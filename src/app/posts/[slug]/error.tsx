"use client"

export default function Error({
  error,
}: {
  error: Error & { digest?: string }
}) {
  console.error(error)

  return (
    <div>
      <h2>Something went wrong!</h2>
    </div>
  )
}
