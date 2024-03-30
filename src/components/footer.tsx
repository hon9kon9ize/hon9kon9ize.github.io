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
        className="flex size-full items-center justify-center"
        href="https://discord.gg/gG6GPp8XxQ"
        target="_blank"
        rel="noreferrer"
      >
        <DiscordLogoIcon className="size-5 text-slate-500 md:size-6" />
      </a>
      <a
        className="flex size-full items-center justify-center"
        href="https://github.com/hon9kon9ize"
        target="_blank"
        rel="noreferrer"
      >
        <GitHubLogoIcon className="size-5 text-slate-500 md:size-6" />
      </a>
      <a
        className="flex size-full items-center justify-center"
        href="https://huggingface.co/hon9kon9ize"
        target="_blank"
        rel="noreferrer"
      >
        <HuggingFaceIcon className="size-5 text-slate-500 md:size-6" />
      </a>
      {/* email */}
      <a
        className="flex size-full items-center justify-center"
        href="mailto:info@hon9kon9ize.com"
      >
        <EnvelopeClosedIcon className="size-5 text-slate-500 md:size-6" />
      </a>
    </div>
  </footer>
)
