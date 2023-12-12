import {
  DiscordLogoIcon,
  EnvelopeClosedIcon,
  GitHubLogoIcon,
} from "@radix-ui/react-icons"

import { HuggingFaceIcon } from "./huggingface-icon"

export const Footer: React.FC = () => (
  <footer className="z-10 flex h-12 w-full flex-col items-center justify-center bg-black/10 md:h-24">
    <div className="flex flex-row gap-4">
      <a
        className="flex h-full w-full items-center justify-center"
        href="https://discord.gg/2pX6Yzj"
        target="_blank"
        rel="noreferrer"
      >
        <DiscordLogoIcon className="h-5 w-5 text-slate-500 md:h-6 md:w-6" />
      </a>
      <a
        className="flex h-full w-full items-center justify-center"
        href="https://github.com/hon9kon9ize"
        target="_blank"
        rel="noreferrer"
      >
        <GitHubLogoIcon className="h-5 w-5 text-slate-500 md:h-6 md:w-6" />
      </a>
      <a
        className="flex h-full w-full items-center justify-center"
        href="https://huggingface.co/hon9kon9ize"
        target="_blank"
        rel="noreferrer"
      >
        <HuggingFaceIcon className="h-5 w-5 text-slate-500 md:h-6 md:w-6" />
      </a>
      {/* email */}
      <a
        className="flex h-full w-full items-center justify-center"
        href="mailto:info@hon9kon9ize.com"
      >
        <EnvelopeClosedIcon className="h-5 w-5 text-slate-500 md:h-6 md:w-6" />
      </a>
    </div>
  </footer>
)
