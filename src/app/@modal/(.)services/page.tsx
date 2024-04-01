"use client"

import { useRouter } from "next/navigation"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  Services,
} from "@/components"

const ServicesModal: React.FC = () => {
  const { back } = useRouter()

  const handleOpenChange = (open: boolean) => {
    console.log(open)

    if (!open) {
      back()
    }
  }

  return (
    <Dialog open onOpenChange={handleOpenChange}>
      <DialogContent>
        <DialogDescription>
          <Services />
        </DialogDescription>
      </DialogContent>
    </Dialog>
  )
}

export default ServicesModal
