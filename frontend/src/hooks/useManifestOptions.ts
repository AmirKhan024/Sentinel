import { useEffect, useRef, useState } from 'react'
import { getManifest, type ManifestComponent } from '../api/meta'
import type { ManifestJson } from '../api/types'
import { classifyError, type ClassifiedError } from '../api/errors'

type ManifestCache = Partial<Record<ManifestComponent, ManifestJson>>

/** Fetches and caches the policy/scheduling manifests once, shared by every ScopeSelector and
 * the Overview page, so navigating between pages doesn't refetch a manifest that never changed
 * within this session. A manual `refetch` is exposed rather than a background revalidation
 * timer, keeping the one fetch fully visible. */
export function useManifestOptions(components: ManifestComponent[]): {
  manifests: ManifestCache
  loading: boolean
  errors: Partial<Record<ManifestComponent, ClassifiedError>>
  refetch: () => void
} {
  const cacheRef = useRef<ManifestCache>({})
  const [manifests, setManifests] = useState<ManifestCache>({})
  const [errors, setErrors] = useState<Partial<Record<ManifestComponent, ClassifiedError>>>({})
  const [loading, setLoading] = useState(false)
  const [generation, setGeneration] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    const toFetch = components.filter((c) => cacheRef.current[c] === undefined || generation > 0)
    if (toFetch.length === 0) {
      setManifests({ ...cacheRef.current })
      return
    }
    setLoading(true)
    Promise.allSettled(toFetch.map((c) => getManifest(c, controller.signal))).then((results) => {
      if (controller.signal.aborted) return
      const nextErrors: Partial<Record<ManifestComponent, ClassifiedError>> = {}
      results.forEach((result, i) => {
        const component = toFetch[i]
        if (result.status === 'fulfilled') {
          cacheRef.current[component] = result.value
        } else {
          nextErrors[component] = classifyError(result.reason)
        }
      })
      setManifests({ ...cacheRef.current })
      setErrors(nextErrors)
      setLoading(false)
    })
    return () => controller.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [components.join(','), generation])

  return { manifests, loading, errors, refetch: () => setGeneration((g) => g + 1) }
}
