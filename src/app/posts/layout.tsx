import Image from "next/image"
import Link from "next/link"

const PostsLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="flex flex-col items-center">
    <Link className="relative flex place-items-center" href="/">
      <Image
        className="relative dark:drop-shadow-[0_0_0.3rem_#ffffff70] dark:invert"
        src="/logo.svg"
        alt="hon9kon9ize logo"
        width={240}
        height={18}
        priority
      />
    </Link>
    {children}
  </div>
)

export default PostsLayout
