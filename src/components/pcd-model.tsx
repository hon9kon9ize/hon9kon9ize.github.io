"use client"

import {
  forwardRef,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import { useFrame, useLoader, useThree } from "@react-three/fiber"
import * as THREE from "three"
import * as TWEEN from "three/examples/jsm/libs/tween.module.js"
import { PCDLoader } from "three/examples/jsm/loaders/PCDLoader.js"

export interface PCDModelProp {
  fileName: string
  pointSize: number
  scale?: THREE.Vector3
  color?: THREE.ColorRepresentation
  rotation?: THREE.Euler
  position?: THREE.Vector3
  mode?: "ring" | "model"
}

function tweenPoints(
  geometry: THREE.BufferGeometry<THREE.NormalBufferAttributes>,
  targetPosition: THREE.TypedArray,
  duration = 1500
) {
  for (let i = 0; i < targetPosition.length / 3; i++) {
    const x = targetPosition[i * 3]
    const y = targetPosition[i * 3 + 1]
    const z = targetPosition[i * 3 + 2]
    const ox = geometry.attributes.position.array[i * 3]
    const oy = geometry.attributes.position.array[i * 3 + 1]
    const oz = geometry.attributes.position.array[i * 3 + 2]

    const fromPoint = { x: ox, y: oy, z: oz }
    const positionAttribute = geometry.getAttribute("position")

    new TWEEN.Tween(fromPoint)
      .to({ x, y, z }, duration)
      .easing(TWEEN.Easing.Quadratic.InOut)
      .delay((duration * i) / targetPosition.length)
      .onUpdate(function () {
        positionAttribute.setXYZ(i, fromPoint.x, fromPoint.y, fromPoint.z)
        positionAttribute.needsUpdate = true // update
      })
      .start()
  }
}

function convertGeometryToRing(
  geometry: THREE.BufferGeometry<THREE.NormalBufferAttributes>,
  radius: number
) {
  const updatedPosition = new Float32Array(geometry.attributes.position.array)
  // initialize the cloud point geometry as a ring at the beginning, use the index to calculate the radius and new position
  for (let i = 0; i < geometry.attributes.position.array.length; i += 3) {
    const index = i / 3
    const angle = (index / 512) * Math.PI * 2

    updatedPosition[i] = radius * Math.cos(angle)
    updatedPosition[i + 1] = radius * Math.sin(angle)
    updatedPosition[i + 2] = 0
  }

  return updatedPosition
}

export const PCDModel = forwardRef<THREE.Mesh, PCDModelProp>(
  ({ fileName, pointSize, scale, color, rotation, position, mode }, ref) => {
    const pointRef = useRef<THREE.Mesh>()
    const { scene } = useThree()
    const data: THREE.Points = useLoader(PCDLoader, fileName)
    const [originalPointPosition, setOriginalPointPosition] =
      useState<THREE.TypedArray>()
    const points = useMemo(() => {
      const pointObject = new THREE.Points(
        data.geometry,
        new THREE.PointsMaterial({ size: pointSize, color: color ?? 0x000000 })
      )

      // save the original point position
      setOriginalPointPosition(data.geometry.attributes.position.array.slice())

      const bbox = new THREE.Box3().setFromObject(data)
      const normalizedScale = 1 / (bbox.max.x - bbox.min.x)
      const initialScale = scale || new THREE.Vector3(1, 1, 1)

      pointObject.scale.set(
        normalizedScale * initialScale.x,
        normalizedScale * initialScale.y,
        normalizedScale * initialScale.z
      )

      return pointObject
    }, [data, pointSize, color, scale])

    // get the min/max of the point position x
    const modelPositionXMinMax = useMemo(
      () =>
        Array.from(data.geometry.attributes.position.array).reduce(
          (acc, cur, index) => {
            if (index % 3 === 0) {
              if (cur < acc[0]) {
                acc[0] = cur
              }

              if (cur > acc[1]) {
                acc[1] = cur
              }
            }

            return acc
          },
          [Infinity, -Infinity]
        ),
      [data.geometry]
    )

    useEffect(() => {
      scene.fog = new THREE.FogExp2(0, 0.065)
    }, [scene])

    useEffect(() => {
      if (!pointRef.current) {
        return
      }

      // convert the point geometry to a ring
      const updatedPositions = convertGeometryToRing(
        pointRef.current.geometry,
        modelPositionXMinMax[1] * 0.05
      )

      pointRef.current.geometry.attributes.position.array.set(updatedPositions)

      pointRef.current.geometry.attributes.position.needsUpdate = true
    }, [modelPositionXMinMax])

    // handle mode
    useEffect(() => {
      if (!pointRef.current) {
        return
      }

      if (mode == "ring") {
        // convert the point geometry to a ring
        const updatedPositions = convertGeometryToRing(
          pointRef.current.geometry,
          modelPositionXMinMax[1] * 0.05
        )

        tweenPoints(pointRef.current.geometry, updatedPositions)

        pointRef.current.geometry.attributes.position.needsUpdate = true
      } else if (mode == "model") {
        // convert the point geometry to a cloud point
        if (originalPointPosition) {
          tweenPoints(pointRef.current.geometry, originalPointPosition)
        }
      }
    }, [mode, modelPositionXMinMax, originalPointPosition])

    useFrame(({ clock }) => {
      const elapsedTime = clock.getElapsedTime()

      if (!pointRef.current) {
        return
      }

      TWEEN.update()

      // if (!isIntroAnimationDone && originalPointPosition) {
      //   // intro animation, move the point from the ring to the original position
      //   const updatedPointPosition = originalPointPosition.map(
      //     (position, index) => {
      //       const updatedPosition = position + 0.00001

      //       return updatedPosition
      //     }
      //   )

      //   pointRef.current.geometry.attributes.position.array.set(
      //     updatedPointPosition
      //   )

      //   pointRef.current.geometry.attributes.position.needsUpdate = true

      //   // if (
      //   //   updatedPointPosition.every(
      //   //     (position, index) => position >= originalPointPosition[index]
      //   //   )
      //   // ) {
      //   //   console.log("d")
      //   //   setIsIntroAnimationDone(true)
      //   // }
      // }

      // zoom in and out by sine wave of the original scale Z
      // const updatedPointScaleDelta = Math.sin(elapsedTime * 0.25) * 1000

      // pointRef.current.scale.set(
      //   pointRef.current.scale.x,
      //   pointRef.current.scale.y,
      //   updatedPointScaleDelta + pointRef.current.scale.z
      // )

      const strideSize = 512
      const strides =
        pointRef.current.geometry.attributes.position.array.length /
        3 /
        strideSize

      for (let i = 0; i < strides; i++) {
        const strideStart = i * strideSize * 3
        const strideEnd = (i + 1) * strideSize * 3

        const stride =
          pointRef.current.geometry.attributes.position.array.slice(
            strideStart,
            strideEnd
          )

        const strideZ = Math.sin(elapsedTime * 0.1) * 0.000001

        for (let j = 0; j < stride.length; j += 3) {
          stride[j + 2] = stride[j + 2] + strideZ
        }

        pointRef.current.geometry.attributes.position.array.set(
          stride,
          strideStart
        )
      }

      pointRef.current.geometry.attributes.position.needsUpdate = true
    })

    return (
      <Suspense fallback={null}>
        {data ? (
          <primitive
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ref={(node: any) => {
              pointRef.current = node

              if (ref) {
                if (typeof ref === "function") {
                  ref(node)
                } else {
                  ref.current = node
                }
              }
            }}
            rotation={rotation || new THREE.Euler(0, 0, 0)}
            position={position || new THREE.Vector3(0, 0, 0)}
            object={points}
          />
        ) : null}
      </Suspense>
    )
  }
)

PCDModel.displayName = "PCDModel"
