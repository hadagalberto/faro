using System.Linq.Expressions;
using System.Reflection;
using Faro.Payload;
using Microsoft.Extensions.VectorData.ProviderServices;
using Microsoft.Extensions.VectorData.ProviderServices.Filter;
using FaroFilter = Faro.Filtering.Filter;

namespace Faro.VectorData;

/// <summary>
/// Translates the supported LINQ predicate subset used by
/// <c>VectorSearchOptions.Filter</c>, <c>HybridSearchOptions.Filter</c> and filtered record
/// retrieval into an immutable Faro filter.
/// </summary>
/// <remarks>
/// Supported: <c>&amp;&amp;</c>, <c>||</c>, <c>!</c>, equality/inequality on keyword, bool and
/// numeric fields, numeric ranges (<c>&gt;</c>, <c>&gt;=</c>, <c>&lt;</c>, <c>&lt;=</c>), a bare
/// bool field, and <c>Contains</c> for tags/keyword fields in either direction. Captured variables
/// and constants are inlined by the abstractions' preprocessing visitor before translation.
/// Everything else throws <see cref="NotSupportedException"/> rather than being silently ignored.
/// </remarks>
internal sealed class FaroFilterTranslator<TRecord> : FilterTranslatorBase
    where TRecord : class
{
    private readonly IReadOnlyDictionary<string, FieldKind> _fields;
    private readonly string _collectionName;

    public FaroFilterTranslator(IReadOnlyDictionary<string, FieldKind> fields, string collectionName)
    {
        _fields = fields;
        _collectionName = collectionName;
    }

    public FaroFilter Translate(Expression<Func<TRecord, bool>> predicate, CollectionModel model)
    {
        ArgumentNullException.ThrowIfNull(predicate);
        Expression body = PreprocessFilter(predicate, model, new FilterPreprocessingOptions
        {
            SupportsParameterization = false,
        });

        return TranslateExpression(body);
    }

    private FaroFilter TranslateExpression(Expression expression)
    {
        expression = StripConvert(expression);
        switch (expression)
        {
            case BinaryExpression binary when binary.NodeType is ExpressionType.AndAlso or ExpressionType.And:
                return FaroFilter.And(TranslateExpression(binary.Left), TranslateExpression(binary.Right));

            case BinaryExpression binary when binary.NodeType is ExpressionType.OrElse or ExpressionType.Or:
                return FaroFilter.Or(TranslateExpression(binary.Left), TranslateExpression(binary.Right));

            case UnaryExpression unary when unary.NodeType == ExpressionType.Not && unary.Operand.Type == typeof(bool):
                return FaroFilter.Not(TranslateExpression(unary.Operand));

            case BinaryExpression comparison when comparison.NodeType is ExpressionType.Equal or ExpressionType.NotEqual:
                return TranslateEquality(comparison);

            case BinaryExpression comparison when comparison.NodeType is
                ExpressionType.GreaterThan or ExpressionType.GreaterThanOrEqual or
                ExpressionType.LessThan or ExpressionType.LessThanOrEqual:
                return TranslateRange(comparison);

            case MethodCallExpression call when call.Method.Name == "Contains":
                return TranslateContains(call);

            case MethodCallExpression call when call.Method.Name == "Equals" && call.Method.DeclaringType == typeof(string):
                return TranslateStringEquals(call);

            case MemberExpression member when member.Type == typeof(bool):
                (string flagField, FieldKind flagKind) = BindField(member);
                if (flagKind != FieldKind.Bool)
                {
                    throw Unsupported($"boolean comparison on field '{flagField}' of kind {flagKind}");
                }

                return FaroFilter.Eq(flagField, true);

            case ConstantExpression constant when constant.Value is bool truth:
                return truth ? FaroFilter.All : FaroFilter.Not(FaroFilter.All);

            default:
                throw Unsupported($"expression '{expression}'");
        }
    }

    private FaroFilter TranslateEquality(BinaryExpression comparison)
    {
        if (TryBindField(comparison.Left, out string? leftField, out FieldKind leftKind))
        {
            return Equality(leftField!, leftKind, EvaluateValue(comparison.Right), comparison.NodeType == ExpressionType.NotEqual);
        }

        if (TryBindField(comparison.Right, out string? rightField, out FieldKind rightKind))
        {
            return Equality(rightField!, rightKind, EvaluateValue(comparison.Left), comparison.NodeType == ExpressionType.NotEqual);
        }

        object? left;
        object? right;
        try
        {
            left = EvaluateValue(comparison.Left);
            right = EvaluateValue(comparison.Right);
        }
        catch (NotSupportedException)
        {
            throw Unsupported($"equality '{comparison}'");
        }

        bool same = Equals(left, right);
        if (comparison.NodeType == ExpressionType.NotEqual)
        {
            same = !same;
        }

        return same ? FaroFilter.All : FaroFilter.Not(FaroFilter.All);
    }

    private static FaroFilter Equality(string field, FieldKind kind, object? value, bool negate)
    {
        FaroFilter result = kind switch
        {
            FieldKind.Keyword or FieldKind.Text => FaroFilter.Eq(field, RequireString(field, value)),
            FieldKind.Bool => FaroFilter.Eq(field, RequireBool(field, value)),
            FieldKind.Numeric => FaroFilter.Eq(field, RequireNumber(field, value)),
            FieldKind.Tag => FaroFilter.Eq(field, RequireString(field, value)),
            _ => throw new NotSupportedException($"Faro cannot compare field '{field}' for equality"),
        };

        return negate ? FaroFilter.Not(result) : result;
    }

    private FaroFilter TranslateRange(BinaryExpression comparison)
    {
        if (!TryBindField(comparison.Left, out string? field, out FieldKind kind) || kind != FieldKind.Numeric)
        {
            throw Unsupported($"range comparison on field '{field ?? "<unbound>"}' (only numeric fields support ranges)");
        }

        double value = RequireNumber(field!, EvaluateValue(comparison.Right));
        return comparison.NodeType switch
        {
            ExpressionType.GreaterThan => FaroFilter.Gt(field!, value),
            ExpressionType.GreaterThanOrEqual => FaroFilter.Ge(field!, value),
            ExpressionType.LessThan => FaroFilter.Lt(field!, value),
            ExpressionType.LessThanOrEqual => FaroFilter.Le(field!, value),
            _ => throw Unsupported($"range operator '{comparison.NodeType}'"),
        };
    }

    private FaroFilter TranslateContains(MethodCallExpression call)
    {
        if (!TryMatchContains(call, out Expression? source, out Expression? item))
        {
            throw Unsupported($"Contains call '{call}'");
        }

        if (TryBindField(source!, out string? sourceField, out FieldKind sourceKind) &&
            sourceKind is FieldKind.Tag or FieldKind.Keyword)
        {
            return FaroFilter.Eq(sourceField!, RequireString(sourceField!, EvaluateValue(item!)));
        }

        if (TryBindField(item!, out string? itemField, out _) && EvaluateStringCollection(source!) is string[] values)
        {
            return FaroFilter.In(itemField!, values);
        }

        throw Unsupported($"Contains call '{call}'");
    }

    private FaroFilter TranslateStringEquals(MethodCallExpression call)
    {
        if (call.Object is not null && TryBindField(call.Object, out string? instanceField, out FieldKind instanceKind) &&
            call.Arguments.Count == 1)
        {
            return Equality(instanceField!, instanceKind, EvaluateValue(call.Arguments[0]), false);
        }

        if (call.Object is null && call.Arguments.Count == 2)
        {
            if (TryBindField(call.Arguments[0], out string? firstField, out FieldKind firstKind))
            {
                return Equality(firstField!, firstKind, EvaluateValue(call.Arguments[1]), false);
            }

            if (TryBindField(call.Arguments[1], out string? secondField, out FieldKind secondKind))
            {
                return Equality(secondField!, secondKind, EvaluateValue(call.Arguments[0]), false);
            }
        }

        throw Unsupported($"string.Equals call '{call}'");
    }

    private (string Field, FieldKind Kind) BindField(Expression expression)
    {
        Expression stripped = StripConvert(expression);
        if (!TryBindProperty(stripped, out PropertyModel? property))
        {
            throw Unsupported($"member access '{expression}' is not a declared record property");
        }

        string field = FaroVectorDataMapping.FieldName(property);
        if (!_fields.TryGetValue(field, out FieldKind kind))
        {
            throw Unsupported($"field '{field}' is not indexed in collection '{_collectionName}'");
        }

        return (field, kind);
    }

    private bool TryBindField(Expression expression, out string? field, out FieldKind kind)
    {
        try
        {
            (field, kind) = BindField(expression);
            return true;
        }
        catch (NotSupportedException)
        {
            field = null;
            kind = default;
            return false;
        }
    }

    private static Expression StripConvert(Expression expression)
    {
        while (expression is UnaryExpression unary &&
               unary.NodeType is ExpressionType.Convert or ExpressionType.ConvertChecked)
        {
            expression = unary.Operand;
        }

        return expression;
    }

    private static object? EvaluateValue(Expression expression)
    {
        expression = StripConvert(expression);
        switch (expression)
        {
            case ConstantExpression constant:
                return constant.Value;
            case QueryParameterExpression parameter:
                return parameter.Value;
            case NewArrayExpression array when array.NodeType == ExpressionType.NewArrayInit:
                var values = new string[array.Expressions.Count];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = RequireString("array", EvaluateValue(array.Expressions[i]));
                }

                return values;
            case MemberExpression member when member.Member is FieldInfo field && field.IsStatic:
                return field.GetValue(null);
            case MemberExpression member when member.Member is PropertyInfo property && property.GetMethod?.IsStatic == true:
                return property.GetValue(null);
            default:
                throw new NotSupportedException($"Faro cannot evaluate filter value expression '{expression}'");
        }
    }

    private static string[]? EvaluateStringCollection(Expression expression)
    {
        object? value = EvaluateValue(expression);
        if (value is string[] array)
        {
            return array;
        }

        if (value is IEnumerable<string> values)
        {
            return values.ToArray();
        }

        return null;
    }

    private static string RequireString(string field, object? value) =>
        value as string ?? throw new NotSupportedException($"field '{field}' requires a string value");

    private static bool RequireBool(string field, object? value) =>
        value is bool flag ? flag : throw new NotSupportedException($"field '{field}' requires a bool value");

    private static double RequireNumber(string field, object? value)
    {
        if (value is null || !FaroModelBuilder.IsNumeric(value.GetType()))
        {
            throw new NotSupportedException($"field '{field}' requires a numeric value");
        }

        return FaroVectorDataMapping.AsNumber(value);
    }

    private NotSupportedException Unsupported(string detail) =>
        new($"Faro filter translation does not support {detail} in collection '{_collectionName}'");
}
