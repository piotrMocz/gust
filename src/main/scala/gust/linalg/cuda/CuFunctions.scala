package gust.linalg.cuda

import gust.util.cuda.{CuContext, CuDevice}
import jcuda.driver._
import java.io.FileWriter
import java.io.BufferedWriter
import scala.util.matching.Regex
import scala.io.Source

/**
 * Created by Piotr on 2014-09-14.
 */
object CuFunctions {

  private def lambda2Cfunc(funcString: String): String = {
    val arrowPattern = "[ ]*=>[ ]*".r
    "__device__ float f(float " + arrowPattern.replaceFirstIn(funcString, ") { return ") + "; }"
  }

  private def appendFuncToFile(funcString: String, filePath: String) {
    // append the functionString to the file (testKernel.cu)
    // (and remove anything after the 12th line :))
    val fileContents = Source.fromFile(filePath).getLines().take(12).mkString("\n") + "\n\n" + lambda2Cfunc(funcString)
    val writer = new BufferedWriter(new FileWriter(filePath, false))

    writer.write(fileContents)
    writer.flush()
    writer.close()
  }

  private def compilePtx(fileName: String): Option[String] = {
    // prepare a path to the .ptx file
    val pattern = "[.]cu$".r
    val ptxName = pattern.replaceFirstIn(fileName, ".ptx")

    // run nvcc:
    val command = "nvcc --ptx -o " + ptxName + " " + fileName
    val proc = Runtime.getRuntime.exec(command)

    if (proc.waitFor() != 0) None
    else Some(ptxName)
  }

  private def launchKernel(A: CuMatrix[Float], funcString: String, ptxName: String): CuMatrix[Float] = {
    JCudaDriver.setExceptionsEnabled(true)

    val blockSize = 256
    // we extend our array so its length is a multiple of blockSize
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val func = new CUfunction()
    JCudaDriver.cuModuleLoad(module, ptxName)

    val funcName = "map_fun"
    JCudaDriver.cuModuleGetFunction(func, module, funcName)

    // kernel parameters:
    val nArr = Array(A.size)
    val n = jcuda.Pointer.to(nArr)

    val params = jcuda.Pointer.to(
      jcuda.Pointer.to(A.offsetPointer),
      n
    )

    val gridDimX = (A.size / blockSize + (if (A.size % blockSize == 0) 0 else 1)) * blockSize

    JCudaDriver.cuLaunchKernel(func,
      gridDimX, 1, 1,
      blockSize, 1, 1,
      0, null, params, null)
    JCudaDriver.cuCtxSynchronize()

    A
  }

  def launchMapFunc(A: CuMatrix[Float], funcString: String): CuMatrix[Float] = {
    val kernelPath = "src/main/resources/gust/linalg/cuda/testKernel.cu"

    // append the function body to the file:
    appendFuncToFile(funcString, kernelPath)
    // compile the file -- TODO some nice exceptions
    compilePtx(kernelPath) match {
      case Some(ptxFile) => launchKernel(A, funcString, ptxFile)
      case None          => println("Ooops"); A
    }
  }

}
