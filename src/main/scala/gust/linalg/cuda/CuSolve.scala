package gust.linalg.cuda

import jcuda.jcublas._
import jcuda.runtime.{cudaMemcpyKind, JCuda}
import spire.syntax.cfor._
import gust.util.cuda
import cuda._


/**
 * Created by piotrek on 05.07.2014.
 */
object CuSolve {

  /**
   * QR solve for matrices that are not necessarily square.
   * The idea is like this (using matlab's notation):
   * We want to solve Ax = b
   * compute QR = A
   * now QRx = b, multiply both sides by Q**-1 (which is Q')
   * we get Rx = Q' * b
   *    [Q, R] = qr(A);
   *    x = R \ (Q' * b);
   *
   * @param A (dimensions: m x n with m >= n)
   * @param b
   * @return
   */
  def QRSolveFloat(A: CuMatrix[Float], b: CuMatrix[Float])(implicit handle: cublasHandle): CuMatrix[Float] = {
    if (A.rows < A.cols) {
      println("Number of rows of matrix A cannot be smaller than the number of columns.")
      return b
    }

    // [Q, R] = qr(A)
    val (d_A, tau) = CuQR.QRFloatMN(A)
    val (d_Q, d_R) = CuQR.QRFactorsFloat(d_A, tau)

    val d_Qtb = d_Q.t * b   // Q' * b

    // solve the triangular system Rx = Q' * b
    JCublas2.cublasStrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, d_R.cols, d_R.offsetPointer, d_R.majorStride, d_Qtb.offsetPointer, 1)

    d_Qtb(0 until A.cols, 0)
  }

  def QRSolveDouble(A: CuMatrix[Double], b: CuMatrix[Double])(implicit handle: cublasHandle): CuMatrix[Double] = {
    if (A.rows < A.cols) {
      println("Number of rows of matrix A cannot be smaller than the number of columns.")
      return b
    }

    // [Q, R] = qr(A)
    val (d_A, tau) = CuQR.QRDoubleMN(A)
    val (d_Q, d_R) = CuQR.QRFactorsDouble(d_A, tau)

    val d_Qtb = d_Q.t * b   // Q' * b

    // solve the triangular system Rx = Q' * b
    JCublas2.cublasDtrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, d_R.cols, d_R.offsetPointer, d_R.majorStride, d_Qtb.offsetPointer, 1)

    d_Qtb(0 until A.cols, 0)
  }



  /** The algorithms below are a Scala-port of the CUDA C version from OrangeOwlSolutions
    * Gauss-Jordan solve of a linear system
    * It is actually quite good for small systems (<= 256**2)
    *
    * The result ends up in the d_B vector */
  def solveSlow(A: CuMatrix[Double], B: CuMatrix[Double])(implicit handle: cublasHandle): CuMatrix[Double] = {
    if (A.rows != B.rows) {
      println("Number of rows in A must be the same as number of rows in B")
      return B
    }

    if (B.cols != 1) {
      println("B has to be a column vector")
      return B
    }

    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return B
    }

    val d_A = CuMatrix.create[Double](A.rows, A.cols)
    val d_B = CuMatrix.create[Double](B.rows, B.cols)
    d_A := A
    d_B := B
    val N = d_A.rows
    // val e_s = A.elemSize   // elems in A and B are the same size

    // this is a workaround for writing:   double alpha;   ..., &alpha   as in C
    val alpha = Array(0.0)
    val beta = Array(0.0)
    val A_ii = Array(0.0)
    val B_i = Array(0.0)
    val a_iiPtr = jcuda.Pointer.to(A_ii)
    val b_iPtr = jcuda.Pointer.to(B_i)
    var cuResult = 0 // just temporary, to check for errors

    val d_R = CuMatrix.zeros[Double](N, 1) // a vector with ratios for every iteration

    for (i <- 0 until (N - 1)) {

      // moving the current elem. on the diagonal to a variable, to use it later
      cuResult = JCuda.cudaMemcpy2D(a_iiPtr, d_A.majorStride * d_A.elemSize,
        d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i, i) * d_A.elemSize),
        d_A.majorStride * d_A.elemSize, d_A.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)


      if (Math.abs(A_ii(0)) < 1e-20) {
        println("A_ii: " + A_ii(0) + ", i: " + i)
        // this could be solved by pivoting...
        println("Division by (nearly) zero.")
        return B
      }

      // copies the whole row "under" the element d_A(i, i) to d_R
      JCublas2.cublasDcopy(handle, N - (i + 1), d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i + 1, i) * d_A.elemSize),
        1, d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i + 1, 0) * d_R.elemSize), 1)


      alpha(0) = 1.0 / A_ii(0)
      // d_R *= alpha
      JCublas2.cublasDscal(handle, N - (i + 1), jcuda.Pointer.to(alpha),
        d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i + 1, 0) * d_R.elemSize), 1)

      JCuda.cudaMemcpy2D(b_iPtr, d_B.majorStride * d_B.elemSize,
        d_B.data.toCuPointer.withByteOffset(d_B.linearIndex(i, 0) * d_B.elemSize),
        d_B.majorStride * d_B.elemSize, d_B.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

      alpha(0) = -B_i(0)
      // d_B += d_R * alpha
      JCublas2.cublasDaxpy(handle, N - (i + 1), jcuda.Pointer.to(alpha),
        d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i + 1, 0) * d_R.elemSize), 1,
        d_B.data.toCuPointer.withByteOffset(d_B.linearIndex(i + 1, 0) * d_B.elemSize), 1)

      alpha(0) = -1.0
      beta(0) = 1.0

      // update of the whole submatrix (col_vector * row_vector == matrix)
      JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
        N - (i + 1), N - i, 1, jcuda.Pointer.to(alpha),
        d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i + 1, 0) * d_R.elemSize), N,
        d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i, i) * d_A.elemSize), N, jcuda.Pointer.to(beta),
        d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i + 1, i) * d_A.elemSize), N)

    }

    // solve the triangular system:
    JCublas2.cublasDtrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, N,
      d_A.data.toCuPointer, d_A.majorSize,
      d_B.data.toCuPointer, 1)

    d_B
  }


  /**
   * Stripe version using cfor for Double values
   * Performance as with whiles, but more readable
   *
   * @param A
   * @param B
   * @param handle
   * @return d_B
   */
  def solveDouble(A: CuMatrix[Double], B: CuMatrix[Double])(implicit handle: cublasHandle): CuMatrix[Double] = {
    if (A.rows != B.rows) {
      println("Number of rows in A must be the same as number of rows in B")
      return B
    }

    if (B.cols != 1) {
      println("B has to be a column vector")
      return B
    }

    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return B
    }

    // copy the matrices:
    val d_A = CuMatrix.create[Double](A.rows, A.cols)
    d_A := A
    //    JCuda.cudaMemcpy2D(d_A.data.toCuPointer, A.majorStride * A.elemSize,
    //      A.data.toCuPointer,
    //      A.majorStride * A.elemSize, A.cols * A.elemSize, A.rows, cudaMemcpyKind.cudaMemcpyDeviceToDevice)

    val d_B = CuMatrix.create[Double](B.rows, B.cols)
    d_B := B
    //    JCuda.cudaMemcpy2D(d_B.data.toCuPointer, B.elemSize,
    //      B.data.toCuPointer,
    //      B.elemSize, B.elemSize, B.rows, cudaMemcpyKind.cudaMemcpyDeviceToDevice)

    val N = d_A.rows
    val matrixBlockSize = 256
    val matrixStripeSize = 128
    val numBlocks = N / matrixBlockSize + (if (N % matrixBlockSize == 0) 0 else 1)

    //val alpha = Array(0.0)
    val beta = Array(0.0)
    val A_ii = Array(0.0)
    val B_i = Array(0.0)
    val a_iiPtr = jcuda.Pointer.to(A_ii)
    val b_iPtr = jcuda.Pointer.to(B_i)
    val one = Array(1.0)
    val onePtr = jcuda.Pointer.to(one)
    val minusOne = Array(-1.0)
    val minusOnePtr = jcuda.Pointer.to(minusOne)

    val d_R = CuMatrix.zeros[Double](N, matrixBlockSize)
    val AMemPointer = d_A.offsetPointer
    val BMemPointer = d_B.offsetPointer
    val RMemPointer = d_R.offsetPointer
    val es = d_A.elemSize

    val vectAii = Array.ofDim[Double](N)
    val vectBi = Array.ofDim[Double](N)

    cfor(0)(_ < numBlocks, _ + 1) { iB => {

      val i0 = iB * N / numBlocks
      val i1 = (iB + 1) * N / numBlocks


      // iB.iB
      cfor(i0)(_ < i1, _ + 1) { i => {
        JCuda.cudaMemcpy2D(a_iiPtr, d_A.majorStride * d_A.elemSize,
          AMemPointer.withByteOffset(d_A.linearIndex(i, i) * es),
          d_A.majorStride * es, d_A.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

        if (Math.abs(A_ii(0)) < 1e-20) {
          println("A_ii: " + A_ii(0) + ", i: " + i)
          println("Division by (nearly) zero.")
          return B
        }

        // cublasDcopy(handle, i1-(i+1), d_A+(i+1)+i*N, 1, d_R+(i+1)+(i-i0)*N, 1)
        JCublas2.cublasDcopy(handle, i1 - (i + 1),
          AMemPointer.withByteOffset(d_A.linearIndex(i + 1, i) * es), 1,
          RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), 1)

        beta(0) = 1.0 / A_ii(0)
        // cublasDscal(handle, i1-(i+1), &beta, d_R+(i+1)+(i-i0)*N, 1)
        JCublas2.cublasDscal(handle, i1 - (i + 1), jcuda.Pointer.to(beta),
          RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), 1)

        JCuda.cudaMemcpy2D(b_iPtr, d_B.majorStride * d_B.elemSize,
          BMemPointer.withByteOffset(d_B.linearIndex(i, 0) * es),
          d_B.majorStride * es, es, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

        beta(0) = -B_i(0)
        //cublasDaxpy(handle, i1-(i+1), &beta, d_R+(i+1)+(i-i0)*N, 1, d_B+(i+1), 1)
        JCublas2.cublasDaxpy(handle, i1 - (i + 1), jcuda.Pointer.to(beta),
          RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), 1,
          BMemPointer.withByteOffset(d_B.linearIndex(i + 1, 0) * es), 1)

        //alpha(0) = -1.0
        //beta(0) = 1.0
        // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i1-(i+1), i1-i, 1, &alpha,
        // d_R+(i+1)+(i-i0)*N, N, d_A+i+i*N, N, &beta, d_A+(i+1)+i*N, N)
        JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, i1 - (i + 1), i1 - i, 1,
          minusOnePtr,
          RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), N,
          AMemPointer.withByteOffset(d_A.linearIndex(i, i) * es), N,
          onePtr,
          AMemPointer.withByteOffset(d_A.linearIndex(i + 1, i) * es), N)

      }
      }

      cfor(iB + 1)(_ < numBlocks, _ + 1) { kB => {
        val k0 = kB * N / numBlocks
        val k1 = (kB + 1) * N / numBlocks

        val numStripes = (i1 - i0) / matrixStripeSize + (if ((i1 - i0) % matrixStripeSize == 0) 0 else 1)

        // for (int iQ=0; iQ<Num_Stripes; iQ++) {
        cfor(0)(_ < numStripes, _ + 1) { iQ => {

          val i_0 = i0 + iQ * (i1 - i0) / numStripes
          val i_1 = i0 + (iQ + 1) * (i1 - i0) / numStripes

          // for (int i=i_0; i<i_1; i++)
          cfor(i_0)(_ < i_1, _ + 1) { i => {
            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i_1-(i+1), k1-k0, 1, &alpha,
            // d_R+(i+1)+(i-i0)*N, N, d_A+i+k0*N, N, &beta, d_A+(i+1)+k0*N, N)
            JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
              i_1 - (i + 1), k1 - k0, 1, minusOnePtr,
              RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), N,
              AMemPointer.withByteOffset(d_A.linearIndex(i, k0) * es), N,
              onePtr,
              AMemPointer.withByteOffset(d_A.linearIndex(i + 1, k0) * es), N)

          }
          }

          // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i1-i_1, k1-k0, i_1-i_0, &alpha,
          // d_R+i_1+(i_0-i0)*N, N, d_A+i_0+k0*N, N, &beta, d_A+i_1+k0*N, N))
          JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
            i1 - i_1, k1 - k0, i_1 - i_0, minusOnePtr,
            RMemPointer.withByteOffset(d_R.linearIndex(i_1, i_0 - i0) * es), N,
            AMemPointer.withByteOffset(d_A.linearIndex(i_0, k0) * es), N,
            onePtr,
            AMemPointer.withByteOffset(d_A.linearIndex(i_1, k0) * es), N)

        }
        }
      }
      }

      if (iB < numBlocks - 1) {
        // double* vectAii = (double*)malloc((i1-i0)*sizeof(double));
        // double* vectBi	= (double*)malloc((i1-i0)*sizeof(double));
        // cublasGetVector(i1-i0, sizeof(double), d_A+i0+i0*N, N+1, vectAii, 1)
        // puts the i1-i0 elems from diagonal in vectAii:
        JCublas2.cublasGetVector(i1 - i0, es.toInt,
          AMemPointer.withByteOffset(d_A.linearIndex(i0, i0) * es), N + 1,
          jcuda.Pointer.to(vectAii), 1)
        // cublasGetVector(i1-i0, sizeof(double), d_B+i0, 1, vectBi, 1)
        JCublas2.cublasGetVector(i1 - i0, es.toInt,
          BMemPointer.withByteOffset(d_B.linearIndex(i0, 0) * es), 1,
          jcuda.Pointer.to(vectBi), 1)


        // jB.iB
        cfor(iB + 1)(_ < numBlocks, _ + 1) { jB => {
          val j0 = jB * N / numBlocks
          val j1 = (jB + 1) * N / numBlocks

          val numStripes = (i1 - i0) / matrixStripeSize + (if ((i1 - i0) % matrixStripeSize == 0) 0 else 1)

          var iQ = 0
          while (iQ < numStripes) {
            val i_0 = i0 + iQ * (i1 - i0) / numStripes
            val i_1 = i0 + (iQ + 1) * (i1 - i0) / numStripes

            var i = i_0
            while (i < i_1) {

              // cublasDcopy(handle, j1-j0, d_A+j0+i*N, 1, d_R+j0+(i-i0)*N, 1)
              JCublas2.cublasDcopy(handle, j1 - j0,
                AMemPointer.withByteOffset(d_A.linearIndex(j0, i) * es), 1,
                RMemPointer.withByteOffset(d_R.linearIndex(j0, i - i0) * es), 1)

              beta(0) = 1.0 / vectAii(i - i0)
              // cublasDscal(handle, j1-j0, &beta, d_R+j0+(i-i0)*N , 1)
              JCublas2.cublasDscal(handle, j1 - j0, jcuda.Pointer.to(beta),
                RMemPointer.withByteOffset(d_R.linearIndex(j0, i - i0) * es), 1)

              beta(0) = -vectBi(i - i0)
              // cublasDaxpy(handle, j1-j0, &beta, d_R+j0+(i-i0)*N, 1, d_B+j0, 1)
              JCublas2.cublasDaxpy(handle, j1 - j0, jcuda.Pointer.to(beta),
                RMemPointer.withByteOffset(d_R.linearIndex(j0, i - i0) * es), 1,
                BMemPointer.withByteOffset(d_B.linearIndex(j0, 0) * es), 1)

              // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, j1-j0, i_1-i, 1, &alpha,
              // d_R+j0+(i-i0)*N, N, d_A+i+i*N, N, &beta, d_A+j0+i*N, N)
              JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
                j1 - j0, i_1 - i, 1, minusOnePtr,
                RMemPointer.withByteOffset(d_R.linearIndex(j0, i - i0) * es), N,
                AMemPointer.withByteOffset(d_A.linearIndex(i, i) * es), N,
                onePtr,
                AMemPointer.withByteOffset(d_A.linearIndex(j0, i) * es), N)

              i += 1
            }

            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, j1-j0, i1-i_1, i_1-i_0, &alpha,
            // d_R+j0+(i_0-i0)*N, N, d_A+i_0+i_1*N, N, &beta, d_A+j0+i_1*N, N)
            JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
              j1 - j0, i1 - i_1, i_1 - i_0, minusOnePtr,
              RMemPointer.withByteOffset(d_R.linearIndex(j0, i_0 - i0) * es), N,
              AMemPointer.withByteOffset(d_A.linearIndex(i_0, i_1) * es), N,
              onePtr,
              AMemPointer.withByteOffset(d_A.linearIndex(j0, i_1) * es), N)


            iQ += 1
          }
        }
        }

        // jB.kB
        // for (int jB = iB + 1; jB < Num_Blocks; jB++) {
        cfor(iB + 1)(_ < numBlocks, _ + 1) { jB => {

          val j0 = jB * N / numBlocks
          val j1 = (jB + 1) * N / numBlocks

          cfor(iB + 1)(_ < numBlocks, _ + 1) { kB => {
            val k0 = kB * N / numBlocks
            val k1 = (kB + 1) * N / numBlocks

            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, j1-j0, k1-k0, i1-i0, &alpha,
            // d_R+j0, N, d_A+i0+k0*N, N, &beta, d_A+j0+k0*N, N)
            JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
              j1 - j0, k1 - k0, i1 - i0, minusOnePtr,
              RMemPointer.withByteOffset(d_R.linearIndex(j0, 0) * es), N,
              AMemPointer.withByteOffset(d_A.linearIndex(i0, k0) * es), N,
              onePtr,
              AMemPointer.withByteOffset(d_A.linearIndex(j0, k0) * es), N)

          }
          }
        }
        }
      }
    }
    }

    // solve the triangular system:
    JCublas2.cublasDtrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, N,
      AMemPointer, d_A.majorSize,
      BMemPointer, 1)

    // the solution vector is in d_B
    d_B
  }


  /**
   * Stripe version using cfor for Float values
   *
   * @param A
   * @param B
   * @param handle
   * @return d_B
   */
  def solveFloat(A: CuMatrix[Float], B: CuMatrix[Float])(implicit handle: cublasHandle): CuMatrix[Float] = {
    if (A.rows != B.rows) {
      println("Number of rows in A must be the same as number of rows in B")
      return B
    }

    if (B.cols != 1) {
      println("B has to be a column vector")
      return B
    }

    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return B
    }

    // copy the matrices:
    val d_A = CuMatrix.create[Float](A.rows, A.cols)
    d_A := A
    //    JCuda.cudaMemcpy2D(d_A.data.toCuPointer, A.majorStride * A.elemSize,
    //      A.data.toCuPointer,
    //      A.majorStride * A.elemSize, A.cols * A.elemSize, A.rows, cudaMemcpyKind.cudaMemcpyDeviceToDevice)

    val d_B = CuMatrix.create[Float](B.rows, B.cols)
    d_B := B
    //    JCuda.cudaMemcpy2D(d_B.data.toCuPointer, B.elemSize,
    //      B.data.toCuPointer,
    //      B.elemSize, B.elemSize, B.rows, cudaMemcpyKind.cudaMemcpyDeviceToDevice)

    val N = d_A.rows
    val matrixBlockSize = 256
    val matrixStripeSize = 128
    val numBlocks = N / matrixBlockSize + (if (N % matrixBlockSize == 0) 0 else 1)

    //val alpha = Array(0.0)
    val beta = Array(0.0f)
    val A_ii = Array(0.0f)
    val B_i = Array(0.0f)
    val a_iiPtr = jcuda.Pointer.to(A_ii)
    val b_iPtr = jcuda.Pointer.to(B_i)
    val one = Array(1.0f)
    val onePtr = jcuda.Pointer.to(one)
    val minusOne = Array(-1.0f)
    val minusOnePtr = jcuda.Pointer.to(minusOne)

    val d_R = CuMatrix.zeros[Float](N, matrixBlockSize)
    val AMemPointer = d_A.offsetPointer
    val BMemPointer = d_B.offsetPointer
    val RMemPointer = d_R.offsetPointer
    val es = d_A.elemSize

    val vectAii = Array.ofDim[Float](N)
    val vectBi = Array.ofDim[Float](N)

    cfor(0)(_ < numBlocks, _ + 1) { iB => {

      val i0 = iB * N / numBlocks
      val i1 = (iB + 1) * N / numBlocks


      // iB.iB
      cfor(i0)(_ < i1, _ + 1) { i => {
        JCuda.cudaMemcpy2D(a_iiPtr, d_A.majorStride * d_A.elemSize,
          AMemPointer.withByteOffset(d_A.linearIndex(i, i) * es),
          d_A.majorStride * es, d_A.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

        if (Math.abs(A_ii(0)) < 1e-20) {
          println("A_ii: " + A_ii(0) + ", i: " + i)
          println("Division by (nearly) zero.")
          return B
        }

        // cublasDcopy(handle, i1-(i+1), d_A+(i+1)+i*N, 1, d_R+(i+1)+(i-i0)*N, 1)
        JCublas2.cublasScopy(handle, i1 - (i + 1),
          AMemPointer.withByteOffset(d_A.linearIndex(i + 1, i) * es), 1,
          RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), 1)

        beta(0) = 1.0f / A_ii(0)
        // cublasDscal(handle, i1-(i+1), &beta, d_R+(i+1)+(i-i0)*N, 1)
        JCublas2.cublasSscal(handle, i1 - (i + 1), jcuda.Pointer.to(beta),
          RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), 1)

        JCuda.cudaMemcpy2D(b_iPtr, d_B.majorStride * d_B.elemSize,
          BMemPointer.withByteOffset(d_B.linearIndex(i, 0) * es),
          d_B.majorStride * es, es, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

        beta(0) = -B_i(0)
        //cublasDaxpy(handle, i1-(i+1), &beta, d_R+(i+1)+(i-i0)*N, 1, d_B+(i+1), 1)
        JCublas2.cublasSaxpy(handle, i1 - (i + 1), jcuda.Pointer.to(beta),
          RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), 1,
          BMemPointer.withByteOffset(d_B.linearIndex(i + 1, 0) * es), 1)

        //alpha(0) = -1.0
        //beta(0) = 1.0
        // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i1-(i+1), i1-i, 1, &alpha,
        // d_R+(i+1)+(i-i0)*N, N, d_A+i+i*N, N, &beta, d_A+(i+1)+i*N, N)
        JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, i1 - (i + 1), i1 - i, 1,
          minusOnePtr,
          RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), N,
          AMemPointer.withByteOffset(d_A.linearIndex(i, i) * es), N,
          onePtr,
          AMemPointer.withByteOffset(d_A.linearIndex(i + 1, i) * es), N)

      }
      }

      cfor(iB + 1)(_ < numBlocks, _ + 1) { kB => {
        val k0 = kB * N / numBlocks
        val k1 = (kB + 1) * N / numBlocks

        val numStripes = (i1 - i0) / matrixStripeSize + (if ((i1 - i0) % matrixStripeSize == 0) 0 else 1)

        // for (int iQ=0; iQ<Num_Stripes; iQ++) {
        cfor(0)(_ < numStripes, _ + 1) { iQ => {

          val i_0 = i0 + iQ * (i1 - i0) / numStripes
          val i_1 = i0 + (iQ + 1) * (i1 - i0) / numStripes

          // for (int i=i_0; i<i_1; i++)
          cfor(i_0)(_ < i_1, _ + 1) { i => {
            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i_1-(i+1), k1-k0, 1, &alpha,
            // d_R+(i+1)+(i-i0)*N, N, d_A+i+k0*N, N, &beta, d_A+(i+1)+k0*N, N)
            JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
              i_1 - (i + 1), k1 - k0, 1, minusOnePtr,
              RMemPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * es), N,
              AMemPointer.withByteOffset(d_A.linearIndex(i, k0) * es), N,
              onePtr,
              AMemPointer.withByteOffset(d_A.linearIndex(i + 1, k0) * es), N)

          }
          }

          // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i1-i_1, k1-k0, i_1-i_0, &alpha,
          // d_R+i_1+(i_0-i0)*N, N, d_A+i_0+k0*N, N, &beta, d_A+i_1+k0*N, N))
          JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
            i1 - i_1, k1 - k0, i_1 - i_0, minusOnePtr,
            RMemPointer.withByteOffset(d_R.linearIndex(i_1, i_0 - i0) * es), N,
            AMemPointer.withByteOffset(d_A.linearIndex(i_0, k0) * es), N,
            onePtr,
            AMemPointer.withByteOffset(d_A.linearIndex(i_1, k0) * es), N)

        }
        }
      }
      }

      if (iB < numBlocks - 1) {
        // double* vectAii = (double*)malloc((i1-i0)*sizeof(double));
        // double* vectBi	= (double*)malloc((i1-i0)*sizeof(double));
        // cublasGetVector(i1-i0, sizeof(double), d_A+i0+i0*N, N+1, vectAii, 1)
        // puts the i1-i0 elems from diagonal in vectAii:
        JCublas2.cublasGetVector(i1 - i0, es.toInt,
          AMemPointer.withByteOffset(d_A.linearIndex(i0, i0) * es), N + 1,
          jcuda.Pointer.to(vectAii), 1)
        // cublasGetVector(i1-i0, sizeof(double), d_B+i0, 1, vectBi, 1)
        JCublas2.cublasGetVector(i1 - i0, es.toInt,
          BMemPointer.withByteOffset(d_B.linearIndex(i0, 0) * es), 1,
          jcuda.Pointer.to(vectBi), 1)


        // jB.iB
        cfor(iB + 1)(_ < numBlocks, _ + 1) { jB => {
          val j0 = jB * N / numBlocks
          val j1 = (jB + 1) * N / numBlocks

          val numStripes = (i1 - i0) / matrixStripeSize + (if ((i1 - i0) % matrixStripeSize == 0) 0 else 1)

          var iQ = 0
          while (iQ < numStripes) {
            val i_0 = i0 + iQ * (i1 - i0) / numStripes
            val i_1 = i0 + (iQ + 1) * (i1 - i0) / numStripes

            var i = i_0
            while (i < i_1) {

              // cublasDcopy(handle, j1-j0, d_A+j0+i*N, 1, d_R+j0+(i-i0)*N, 1)
              JCublas2.cublasScopy(handle, j1 - j0,
                AMemPointer.withByteOffset(d_A.linearIndex(j0, i) * es), 1,
                RMemPointer.withByteOffset(d_R.linearIndex(j0, i - i0) * es), 1)

              beta(0) = 1.0f / vectAii(i - i0)
              // cublasDscal(handle, j1-j0, &beta, d_R+j0+(i-i0)*N , 1)
              JCublas2.cublasSscal(handle, j1 - j0, jcuda.Pointer.to(beta),
                RMemPointer.withByteOffset(d_R.linearIndex(j0, i - i0) * es), 1)

              beta(0) = -vectBi(i - i0)
              // cublasDaxpy(handle, j1-j0, &beta, d_R+j0+(i-i0)*N, 1, d_B+j0, 1)
              JCublas2.cublasSaxpy(handle, j1 - j0, jcuda.Pointer.to(beta),
                RMemPointer.withByteOffset(d_R.linearIndex(j0, i - i0) * es), 1,
                BMemPointer.withByteOffset(d_B.linearIndex(j0, 0) * es), 1)

              // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, j1-j0, i_1-i, 1, &alpha,
              // d_R+j0+(i-i0)*N, N, d_A+i+i*N, N, &beta, d_A+j0+i*N, N)
              JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
                j1 - j0, i_1 - i, 1, minusOnePtr,
                RMemPointer.withByteOffset(d_R.linearIndex(j0, i - i0) * es), N,
                AMemPointer.withByteOffset(d_A.linearIndex(i, i) * es), N,
                onePtr,
                AMemPointer.withByteOffset(d_A.linearIndex(j0, i) * es), N)

              i += 1
            }

            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, j1-j0, i1-i_1, i_1-i_0, &alpha,
            // d_R+j0+(i_0-i0)*N, N, d_A+i_0+i_1*N, N, &beta, d_A+j0+i_1*N, N)
            JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
              j1 - j0, i1 - i_1, i_1 - i_0, minusOnePtr,
              RMemPointer.withByteOffset(d_R.linearIndex(j0, i_0 - i0) * es), N,
              AMemPointer.withByteOffset(d_A.linearIndex(i_0, i_1) * es), N,
              onePtr,
              AMemPointer.withByteOffset(d_A.linearIndex(j0, i_1) * es), N)


            iQ += 1
          }
        }
        }

        // jB.kB
        // for (int jB = iB + 1; jB < Num_Blocks; jB++) {
        cfor(iB + 1)(_ < numBlocks, _ + 1) { jB => {

          val j0 = jB * N / numBlocks
          val j1 = (jB + 1) * N / numBlocks

          cfor(iB + 1)(_ < numBlocks, _ + 1) { kB => {
            val k0 = kB * N / numBlocks
            val k1 = (kB + 1) * N / numBlocks

            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, j1-j0, k1-k0, i1-i0, &alpha,
            // d_R+j0, N, d_A+i0+k0*N, N, &beta, d_A+j0+k0*N, N)
            JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
              j1 - j0, k1 - k0, i1 - i0, minusOnePtr,
              RMemPointer.withByteOffset(d_R.linearIndex(j0, 0) * es), N,
              AMemPointer.withByteOffset(d_A.linearIndex(i0, k0) * es), N,
              onePtr,
              AMemPointer.withByteOffset(d_A.linearIndex(j0, k0) * es), N)

          }
          }
        }
        }
      }
    }
    }

    // solve the triangular system:
    JCublas2.cublasStrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, N,
      AMemPointer, d_A.majorSize,
      BMemPointer, 1)

    // the solution vector is in d_B
    d_B
  }




}