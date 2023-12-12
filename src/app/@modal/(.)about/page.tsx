"use client"

import { useRouter } from "next/navigation"
import { About, Dialog, DialogContent, DialogDescription } from "@/components"

const AboutModal: React.FC = () => {
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
          <About />
        </DialogDescription>
      </DialogContent>
    </Dialog>
  )
}

export default AboutModal
