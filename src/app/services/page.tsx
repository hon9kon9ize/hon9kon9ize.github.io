import { Metadata } from "next"
import { Services, SERVICES_SUMMARY } from "@/components"

export const metadata: Metadata = {
  title: "Services | hon9kon9ize",
  description: SERVICES_SUMMARY,
}

const ServicesPage: React.FC = () => (
  <div className="flex-col p-12 md:p-24">
    <Services />
  </div>
)

export default ServicesPage
