/**
 * The name and address of an establishment, with its internal id demoted to a small secondary
 * line. Every list page used to show only `establishment_id` (e.g. "EST-00002282595") because
 * nothing downstream of Component 2's entity resolution ever read the name/address it already
 * computes -- see `sentinel.api.services.entity_service`. This is the one place that renders an
 * establishment's identity, so every table and page header looks the same and degrades the same
 * way when a name genuinely isn't available (never invents one).
 */
export function EstablishmentIdentity({
  name,
  address,
  establishmentId,
}: {
  name: string | null | undefined
  address: string | null | undefined
  establishmentId: string
}) {
  const hasName = Boolean(name && name.trim().length > 0)
  return (
    <span className="establishment-identity">
      <span className="establishment-identity-name">{hasName ? name : establishmentId}</span>
      {address && <span className="establishment-identity-address">{address.trim()}</span>}
      {hasName && <span className="establishment-identity-id">{establishmentId}</span>}
    </span>
  )
}
