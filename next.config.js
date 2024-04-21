/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  reactStrictMode: true,
  images: { unoptimized: true },
  env: {
    NEXT_PUBLIC_MEASUREMENT_ID: 'G-LC1QPN7WB3',
  }
}

module.exports = nextConfig
