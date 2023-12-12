import type { Metadata } from "next"
import { Space_Mono } from "next/font/google"

import "./globals.css"

import { Footer } from "@/components/footer"

const inter = Space_Mono({ weight: "400", subsets: ["latin"] })

export const metadata: Metadata = {
  title: "hon9kon9ize",
  description: "Hongkongizing the world through technology",
}

export default function RootLayout({
  children,
  modal,
}: {
  children: React.ReactNode
  modal: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex min-h-screen flex-col">
          <main className="flex flex-1 flex-col p-8 md:p-24">{children}</main>
          <Footer />
        </div>
        {modal}
      </body>
    </html>
  )
}
