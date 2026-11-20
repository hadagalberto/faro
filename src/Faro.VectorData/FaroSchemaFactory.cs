using System.Globalization;
using Faro.Payload;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace Faro.VectorData;

/// <summary>
/// Translates the abstractions' record model into the delegate-based (AOT-safe) Faro payload
/// schema. Property access goes through the abstractions' <see cref="PropertyModel"/> accessors,
/// so both attribute-annotated POCOs and explicit definitions work identically.
/// </summary>
internal static class FaroSchemaFactory
{
    public static PayloadSchema<TRecord> Create<TRecord>(CollectionModel model, string collectionName)
        where TRecord : class
    {
        var schema = new PayloadSchema<TRecord>();

        foreach (DataPropertyModel property in model.DataProperties)
        {
            string name = FaroVectorDataMapping.FieldName(property);
            if (property.IsFullTextIndexed)
            {
                schema.FullText(name, record => property.GetValueAsObject(record) as string ?? string.Empty);
                continue;
            }

            if (!property.IsIndexed)
            {
                continue;
            }

            FieldKind kind = FaroVectorDataMapping.FieldKindFor(property.Type);
            switch (kind)
            {
                case FieldKind.Keyword:
                    schema.Keyword(name, record => property.GetValueAsObject(record) as string ?? string.Empty);
                    break;
                case FieldKind.Numeric:
                    schema.Numeric(name, record => FaroVectorDataMapping.AsNumber(property.GetValueAsObject(record)));
                    break;
                case FieldKind.Bool:
                    schema.Bool(name, record => property.GetValueAsObject(record) is bool value && value);
                    break;
                case FieldKind.Tag:
                    schema.Tags(name, record => (property.GetValueAsObject(record) as IEnumerable<string>) ?? []);
                    break;
                default:
                    throw FaroVectorDataMapping.Unsupported(collectionName, $"field kind '{kind}'");
            }
        }

        // Keys are stored separately by Faro but are also exposed as an indexed keyword field so
        // filters such as r => r.Id == "abc" can be translated and evaluated by the payload index.
        foreach (KeyPropertyModel property in model.KeyProperties)
        {
            string name = FaroVectorDataMapping.FieldName(property);
            if (schema.FieldNames.Contains(name))
            {
                continue;
            }

            schema.Keyword(name, record => Convert.ToString(
                property.GetValueAsObject(record), CultureInfo.InvariantCulture) ?? string.Empty);
        }

        if (schema.FieldNames.Count == 0)
        {
            throw FaroVectorDataMapping.Unsupported(
                collectionName, "a record with no indexed data, full-text or key field");
        }

        return schema;
    }
}
