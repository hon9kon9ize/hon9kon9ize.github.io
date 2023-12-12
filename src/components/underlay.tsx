"use client"

import { useRef } from "react"
import { Stats } from "@react-three/drei"
import { Canvas } from "@react-three/fiber"
import * as THREE from "three"

import { PCDModel } from "@/components/pcd-model"

const pointCloudScale = new THREE.Vector3(2, -2, 2)
const isDev = process.env.NODE_ENV !== "production"

export const Underlay: React.FC<{
  mode?: "ring" | "model"
}> = ({ mode = "model" }) => {
  const pointRef = useRef<THREE.Mesh>(null)

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        position: "fixed",
        top: 0,
        left: 0,
        zIndex: 0,
      }}
    >
      <Canvas
        shadows={false}
        gl={{ alpha: true, stencil: false, depth: false, antialias: false }}
        camera={{ position: [0, 0, 18], fov: 2, near: 1, far: 100 }}
      >
        <PCDModel
          ref={pointRef}
          fileName="/hk.pcd"
          pointSize={0.02}
          color="#ccc"
          mode={mode}
          scale={pointCloudScale}
          // position={new THREE.Vector3(0, 0, 0)}
          // rotation={new THREE.Euler(-0.2, 0, 0)}
        />
        {isDev && <Stats />}
      </Canvas>
    </div>
  )
}
