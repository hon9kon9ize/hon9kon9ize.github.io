import { Metadata } from "next"
import { About } from "@/components"

export const metadata: Metadata = {
  title: "About | hon9kon9ize",
  description:
    "We are a dedicated group of individuals who share a deep passion for Cantonese culture, language, and the field of machine learning.",
}

const AboutPage: React.FC = () => (
  <div className="flex-col p-12 md:p-24">
    <About />
  </div>
)

export default AboutPage
