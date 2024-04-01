"use client"

import { forwardRef } from "react"
import Image from "next/image"
import Link from "next/link"
import { Dialog, Underlay } from "@/components"
import { clsx } from "clsx"

const NavItemLink = forwardRef<
  HTMLAnchorElement,
  {
    children: React.ReactNode
    asChild?: boolean
  } & React.AnchorHTMLAttributes<HTMLAnchorElement>
>(({ children, ...props }, ref) => {
  return (
    <a
      ref={ref}
      {...props}
      className={clsx(
        "group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-400 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30",
        props.className
      )}
    >
      {children}
    </a>
  )
})

NavItemLink.displayName = "NavItemLink"

const ItemArrow: React.FC = () => (
  <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
    -&gt;
  </span>
)

const Overlay: React.FC = () => {
  return (
    <div className="relative z-10 flex flex-1 flex-col items-center justify-between">
      <div className="relative flex flex-1 place-items-center">
        <div className="flex flex-col items-center justify-between gap-4">
          <Image
            className="relative dark:drop-shadow-[0_0_0.3rem_#ffffff70] dark:invert"
            src="/logo.svg"
            alt="hon9kon9ize logo"
            width={420}
            height={32}
            style={{
              width: "100%",
              height: "auto",
            }}
            sizes="100vw"
            priority
          />
          <p className="text-gray-500">AI Lab</p>
        </div>
      </div>

      <div className="mb-32 mt-8 grid justify-center text-center md:mt-0 lg:mb-0 lg:w-full lg:max-w-5xl lg:grid-cols-4 lg:text-left">
        <Dialog />
        <Link legacyBehavior href="/about">
          <NavItemLink className="cursor-pointer">
            <h2 className={`mb-3 text-2xl font-semibold`}>
              About Us <ItemArrow />
            </h2>
            <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
              We are a group of people who are passionate about Cantonese
              culture and language.
            </p>
          </NavItemLink>
        </Link>

        <Link legacyBehavior href="/services">
          <NavItemLink className="cursor-pointer">
            <h2 className={`mb-3 text-2xl font-semibold`}>
              Services <ItemArrow />
            </h2>
            <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
              We provide model training and consulting services
            </p>
          </NavItemLink>
        </Link>

        <Link legacyBehavior passHref href="https://huggingface.co/hon9kon9ize">
          <NavItemLink target="_blank" className="cursor-pointer">
            <h2 className={`mb-3 text-2xl font-semibold`}>
              CantoneseLLM <ItemArrow />
            </h2>
            <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
              CantoneseLLM is a language model family that specifically intended
              for Cantonese language.
            </p>
          </NavItemLink>
        </Link>

        <Link href="/posts" legacyBehavior>
          <NavItemLink className="cursor-pointer">
            <h2 className={`mb-3 text-2xl font-semibold`}>
              Blog <ItemArrow />
            </h2>
            <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
              Some thoughts about Machine Learning and Cantonese.
            </p>
          </NavItemLink>
        </Link>
      </div>
    </div>
  )
}

export default function Home() {
  return (
    <>
      <Underlay mode="model" />
      <Overlay />
    </>
  )
}
