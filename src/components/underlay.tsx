"use client"

import { useEffect, useRef, useState } from "react"
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
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const gl = useRef<WebGLRenderingContext>()
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    try{
      gl.current = new THREE.WebGLRenderer({ canvas: canvasRef.current!,
        antialias: false,
        stencil: false,
        depth: false,
        alpha: true,
      }).getContext();
    } catch (e) {
      setError(e as Error)
    }
  }, [canvasRef])

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
      {!error && (
      <Canvas
        ref={canvasRef}
        shadows={false}
        gl={gl.current}
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
      )}
    </div>
  )
}
