#include <stdio.h>
#include <cuda_runtime.h>

// Lower/upper point limits for dudt
#define LP -1
#define UP 1

// Perpendicular current cell  limits
#define P0 -1
#define P1 1

__forceinline__ __device__ int signbit_custom( const double x ) {

  if ( x < 0 ) {
    return 1;
  } else {
    return 0;
  }

}

__forceinline__ __device__ void spline_s2( const double x, double *const s ) {

  const double t0 = 0.5 - x;
  const double t1 = 0.5 + x;

  s[-1] = 0.5 * t0*t0;
  s[ 0] = 0.5 + t0*t1;
  s[ 1] = 0.5 * t1*t1;

}

__forceinline__ __device__ void splineh_s2( const double x, const int h,
                                            double *const s ) {

  const double t0 = ( 1 - h ) - x;
  const double t1 = (     h ) + x;

  s[-1] = 0.5 * t0*t0;
  s[ 0] = 0.5 + t0*t1;
  s[ 1] = 0.5 * t1*t1;

}

__forceinline__ __device__ void wl_s2( const double qnx, const double x0,
                                       const double x1, double *const wl ) {

  const double d    = x1 - x0;
  const double s1_2 = 0.5 - x0;
  const double p1_2 = 0.5 + x0;

  wl[-1] = s1_2 - 0.5 * d;
  wl[ 0] = p1_2 + 0.5 * d;

  const double n = qnx * d;
  wl[-1] *= n;
  wl[ 0]  *= n;

}

__forceinline__ __device__ int ntrim( const double x ) {
  int a, b;

  if ( x < -.5 ) {
    a = -1;
  } else {
    a = 0;
  }

  if ( x >= .5 ) {
    b = +1;
  } else {
    b = 0;
  }

  return a + b;

}

template <int BLOCK_SIZE>
__device__ void prescan(int *const idata) {

  int tx = threadIdx.x;
  int offset = 1;

  // build sum in place up the tree
  for (int d = BLOCK_SIZE>>1; d > 0; d >>= 1) {
    __syncthreads();
    if (tx < d) {
      int ai = offset*(2*tx+1)-1;
      int bi = offset*(2*tx+2)-1;
      idata[bi] += idata[ai];
    }
    offset *= 2;
  }

  // clear the last element
  if (tx == 0) {
    idata[BLOCK_SIZE - 1] = 0;
  }

  // traverse down tree & build scan
  for (int d = 1; d < BLOCK_SIZE; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (tx < d) {
      int ai = offset*(2*tx+1)-1;
      int bi = offset*(2*tx+2)-1;
      int t = idata[ai];
      idata[ai] = idata[bi];
      idata[bi] += t;
    }
  }

}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

template <int BLOCK_SIZE>
__global__ void DUDT_BORIS_2D( char *chunks, int nchunks_per_block, int nchunks, int ntils, int npart,
                               int stride, double rqm, double **b_f2,
                               double **e_f2, double dt, int chunk_size ){

  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int bx_tot = blockDim.x;

  const size_t x_offset = 0;
  const size_t p_offset = x_offset + 2 * sizeof(double) * chunk_size;
  const size_t q_offset = p_offset + 3 * sizeof(double) * chunk_size;
  const size_t ix_offset = q_offset + sizeof(double) * chunk_size;
  const size_t chunk_mem = ix_offset + 2 * sizeof(int) * chunk_size;

  int nblocks_per_tile = bx_tot / ntils;
  int til = bx / nblocks_per_tile;
  int block = bx - til * nblocks_per_tile;
  int num_par_proc_tot = nchunks_per_block * chunk_size * (block+1<nblocks_per_tile) +
                         ( npart - (nblocks_per_tile-1) * nchunks_per_block * chunk_size ) *
                         (block+1==nblocks_per_tile);
  char *chunk = chunks + (block*nchunks_per_block + til*nchunks) * chunk_mem;

  double *b_f2_til = b_f2[til];
  double *e_f2_til = e_f2[til];

  // Shared variables that are the same between all threads
  double tem = 0.5 * dt / rqm;

  int i1, i2, i1h, i2h, h1, h2, num_par_proc_chunk, part_idx;

  double u2, gamma, gam_tem, otsq;
  double bp[3], ep[3], utemp[3];

  double dx1, dx2;
  double f1, f2, f3, f1line, f2line, f3line;

  double w1_buff[UP-LP+1], w2_buff[UP-LP+1], w1h_buff[UP-LP+1], w2h_buff[UP-LP+1];

  double *const w1  = w1_buff  - LP;
  double *const w2  = w2_buff  - LP;
  double *const w1h = w1h_buff - LP;
  double *const w2h = w2h_buff - LP;

  while( num_par_proc_tot > 0 ) {

    double *x = (double*) ( chunk + x_offset );
    double *p = (double*) ( chunk + p_offset );
    // double *q = (double*) ( chunk + q_offset );
    int   *ix = (int *)   ( chunk + ix_offset);

    // modify bp & ep to include timestep and charge-to-mass ratio
    // and perform half the electric field acceleration.
    // Result is stored in UTEMP.

    // loop over chunks of particles until all particles in tile have been processed
    num_par_proc_chunk = min(num_par_proc_tot, chunk_size);
    part_idx = tx;

    while ( part_idx < num_par_proc_chunk ){

      // --------------
      // dudt_boris()
      // --------------
      i1 = ix[part_idx+0*chunk_size] - 1;
      i2 = ix[part_idx+1*chunk_size] - 1;

      dx1 = x[part_idx+0*chunk_size];
      dx2 = x[part_idx+1*chunk_size];

      h1 = signbit_custom(dx1);
      h2 = signbit_custom(dx2);

      i1h = i1 - h1;
      i2h = i2 - h2;

      // get spline weitghts for x and y
      spline_s2( dx1, w1 );
      splineh_s2( dx1, h1, w1h );

      spline_s2( dx2, w2 );
      splineh_s2( dx2, h2, w2h );

      // Interpolate E - Field
      f1 = 0;
      f2 = 0;
      f3 = 0;

      for (int k2=LP; k2<=UP; ++k2){
        f1line = 0;
        f2line = 0;
        f3line = 0;

        for (int k1=LP; k1<=UP; ++k1){
          f1line += e_f2_til[0+3*(i1h+k1+(i2 +k2)*stride)] * w1h[k1];
          f2line += e_f2_til[1+3*(i1 +k1+(i2h+k2)*stride)] * w1[ k1];
          f3line += e_f2_til[2+3*(i1 +k1+(i2 +k2)*stride)] * w1[ k1];
        }

        f1 += f1line * w2[ k2];
        f2 += f2line * w2h[k2];
        f3 += f3line * w2[ k2];
      }

      ep[0] = f1;
      ep[1] = f2;
      ep[2] = f3;

      // Interpolate B - Field
      f1 = 0;
      f2 = 0;
      f3 = 0;

      for (int k2=LP; k2<=UP; ++k2){
        f1line = 0;
        f2line = 0;
        f3line = 0;

        for (int k1=LP; k1<=UP; ++k1){
          f1line += b_f2_til[0+3*(i1 +k1+(i2h+k2)*stride)] * w1[ k1];
          f2line += b_f2_til[1+3*(i1h+k1+(i2 +k2)*stride)] * w1h[k1];
          f3line += b_f2_til[2+3*(i1h+k1+(i2h+k2)*stride)] * w1h[k1];
        }

        f1 += f1line * w2h[k2];
        f2 += f2line * w2[ k2];
        f3 += f3line * w2h[k2];
      }

      bp[0] = f1;
      bp[1] = f2;
      bp[2] = f3;

      // --------------

      ep[0] *= tem;
      ep[1] *= tem;
      ep[2] *= tem;

      utemp[0] = p[part_idx+0*chunk_size] + ep[0];
      utemp[1] = p[part_idx+1*chunk_size] + ep[1];
      utemp[2] = p[part_idx+2*chunk_size] + ep[2];

      // Get time centered gamma
      u2 = utemp[0]*utemp[0] + utemp[1]*utemp[1] + utemp[2]*utemp[2];

      gamma = sqrt(u2+1);

      gam_tem = tem / gamma;

      bp[0] *= gam_tem;
      bp[1] *= gam_tem;
      bp[2] *= gam_tem;

      p[part_idx+0*chunk_size] = utemp[0] + utemp[1] * bp[2];
      p[part_idx+1*chunk_size] = utemp[1] + utemp[2] * bp[0];
      p[part_idx+2*chunk_size] = utemp[2] + utemp[0] * bp[1];

      p[part_idx+0*chunk_size] -= utemp[2] * bp[1];
      p[part_idx+1*chunk_size] -= utemp[0] * bp[2];
      p[part_idx+2*chunk_size] -= utemp[1] * bp[0];

      otsq = 2.0 / ( ( (1.0 + bp[0]*bp[0]) + bp[1]*bp[1] ) + bp[2]*bp[2] );

      bp[0] *= otsq;
      bp[1] *= otsq;
      bp[2] *= otsq;

      utemp[0] += p[part_idx+1*chunk_size] * bp[2];
      utemp[1] += p[part_idx+2*chunk_size] * bp[0];
      utemp[2] += p[part_idx+0*chunk_size] * bp[1];

      utemp[0] -= p[part_idx+2*chunk_size] * bp[1];
      utemp[1] -= p[part_idx+0*chunk_size] * bp[2];
      utemp[2] -= p[part_idx+1*chunk_size] * bp[0];

      // Perform second half of electric field acceleration.
      p[part_idx+0*chunk_size] = utemp[0] + ep[0];
      p[part_idx+1*chunk_size] = utemp[1] + ep[1];
      p[part_idx+2*chunk_size] = utemp[2] + ep[2];

      // TODO note, this only works if the chunks size is divisible by the block size
      // see sort for how to do it more generally. Maybe we want to do that here too
      part_idx += BLOCK_SIZE;

    } // end loop over chunk of particles ``while ( part_idx < num_par )``

    num_par_proc_tot -= num_par_proc_chunk;
    chunk += chunk_mem;

  } // end loop over chunks

}

template <int BLOCK_SIZE>
__global__ void ADVANCE_DEPOSIT_2D( char *chunks, int nchunks_per_block, int nchunks, int ntils, int npart,
                                    int stride, double dx1, double dx2,
                                    double **jay_f2, double dt,
                                    int chunk_size ){

  // Possible number of virtual particles
#define NP 3

  // Block, thread index
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int bx_tot = blockDim.x;

  const size_t x_offset = 0;
  const size_t p_offset = x_offset + 2 * sizeof(double) * chunk_size;
  const size_t q_offset = p_offset + 3 * sizeof(double) * chunk_size;
  const size_t ix_offset = q_offset + sizeof(double) * chunk_size;
  const size_t chunk_mem = ix_offset + 2 * sizeof(int) * chunk_size;

  int nblocks_per_tile = bx_tot / ntils;
  int til = bx / nblocks_per_tile;
  int block = bx - til * nblocks_per_tile;
  int num_par_proc_tot = nchunks_per_block * chunk_size * (block+1<nblocks_per_tile) +
                         ( npart - (nblocks_per_tile-1) * nchunks_per_block * chunk_size ) *
                         (block+1==nblocks_per_tile);
  char *chunk = chunks + (block*nchunks_per_block + til*nchunks) * chunk_mem;

  // Shared variables that will be scanned between threads
  __shared__ int num_par[NP*BLOCK_SIZE];
  __shared__ int ii[NP*BLOCK_SIZE], jj[NP*BLOCK_SIZE];
  __shared__ double x0[NP*BLOCK_SIZE], y0[NP*BLOCK_SIZE], qq[NP*BLOCK_SIZE];
  __shared__ double x1[NP*BLOCK_SIZE], y1[NP*BLOCK_SIZE], vz[NP*BLOCK_SIZE];

  double *jay_f2_til = jay_f2[til];

  double dt_dx1 = dt/dx1;
  double dt_dx2 = dt/dx2;

  double rgamma;
  double xbuf[2];

  int dxi[2], dxi_fac_x[2], dxi_fac_i[2], part_idx, num_par_proc_chunk;
  bool no_y_cross;

  int p_ind[NP], ind, num_par_tot[NP], posneg[2], i1, i2;

  double xint[4], delta[2], vz_, vzint[NP];

  double qnx, qny, qvz;
  double tmp1, tmp2;

  double S0x_buff[P1-P0+1], S1x_buff[P1-P0+1], S0y_buff[P1-P0+1], S1y_buff[P1-P0+1];
  double wp1_buff[P1-P0+1], wp2_buff[P1-P0+1];
  double wl1_buff[P1-P0+1], wl2_buff[P1-P0+1];

  double *const S0x = S0x_buff - P0;
  double *const S1x = S1x_buff - P0;
  double *const S0y = S0y_buff - P0;
  double *const S1y = S1y_buff - P0;
  double *const wp1 = wp1_buff - P0;
  double *const wp2 = wp2_buff - P0;
  double *const wl1 = wl1_buff - P0;
  double *const wl2 = wl2_buff - P0;

  const double c1_3 = 1.0 / 3.0;
  const double jnorm1 = dx1/dt/2;
  const double jnorm2 = dx2/dt/2;

  // loop over chunks of particles until all particles in tile have been processed
  while( num_par_proc_tot > 0 ) {

    double *x = (double*) ( chunk + x_offset );
    double *p = (double*) ( chunk + p_offset );
    double *q = (double*) ( chunk + q_offset );
    int   *ix = (int *)   ( chunk + ix_offset);

    num_par_proc_chunk = min(num_par_proc_tot, chunk_size);
    part_idx = tx;
    int n_chunk_iters = (num_par_proc_chunk+BLOCK_SIZE-1)/BLOCK_SIZE;

    // loop over particles in a chunk, BLOCK_SIZE at a time
    for ( int j=0; j<n_chunk_iters; j++ ) {

      if ( part_idx < num_par_proc_chunk ) {

        rgamma = rsqrt( ( (1.0 + p[part_idx+0*chunk_size]*p[part_idx+0*chunk_size] )
                                         + p[part_idx+1*chunk_size]*p[part_idx+1*chunk_size] )
                                         + p[part_idx+2*chunk_size]*p[part_idx+2*chunk_size] );

        xbuf[0] = x[part_idx+0*chunk_size] + ( p[part_idx+0*chunk_size] * rgamma ) * dt_dx1;
        xbuf[1] = x[part_idx+1*chunk_size] + ( p[part_idx+1*chunk_size] * rgamma ) * dt_dx2;

      } else {

        xbuf[0] = 0;
        xbuf[1] = 0;

      }

      dxi[0] = ntrim(xbuf[0]);
      dxi[1] = ntrim(xbuf[1]);

      // --------------
      // dep_current_2d_s*()
      // --------------

      // --------------
      // split_2d()
      // --------------

      // Splits particle trajectories so that all virtual particles have a motion starting and ending
      // in the same cell. This routines also calculates vz for each virtual particle.

      // create virtual particles that correspond to a motion that starts and ends
      // in the same grid cell

      // For switch case, -1 if no particle
      int cross = abs(dxi[0]) + 2 * abs(dxi[1]) - (part_idx>=num_par_proc_chunk);
      // number of virtual particles for this thread
      int my_num_par = (cross + 1) / 2 + 1 - (cross<0);

      for ( int l=0; l<NP; l++ ) {
        num_par[tx+l*BLOCK_SIZE] = (my_num_par>l);
      }
      __syncthreads();

      for ( int l=0; l<NP; l++ ) {
        num_par_tot[l] = num_par[(l+1)*BLOCK_SIZE-1];
      }

      prescan<BLOCK_SIZE>( num_par );
      prescan<BLOCK_SIZE>( num_par+  BLOCK_SIZE );
      prescan<BLOCK_SIZE>( num_par+2*BLOCK_SIZE );
      __syncthreads();

      for ( int l=0; l<NP; l++ ) {
        num_par_tot[l] += num_par[(l+1)*BLOCK_SIZE-1];
      }

      int k = 0;
      for ( int l=0; l<NP; l++ ) {
        p_ind[l] = num_par[tx+l*BLOCK_SIZE] + k;
        k += num_par_tot[l];
      }

      if (my_num_par>0) {

        vz_ = p[part_idx+2*chunk_size]*rgamma;
        vzint[0] = vz_;
        vzint[1] = vz_;
        vzint[2] = vz_;
        no_y_cross = true;

        if (cross==0) {
          // no cross
          xint[0] = xbuf[0];
          xint[1] = xbuf[1];
          delta[0] = 1.0;

          // vzint[0] = vz_;

        } else if (cross<3) {
          // single cross over x or y
          i1 = cross-1; i2 = 2-cross;
          xint[i1]  = 0.5 * dxi[i1];
          delta[0]  = ( xint[i1] - x[part_idx+i1*chunk_size] ) / ( xbuf[i1] - x[part_idx+i1*chunk_size] );
          xint[i2]  =  x[part_idx+i2*chunk_size] + (xbuf[i2] - x[part_idx+i2*chunk_size]) * delta[0];

          xint[2] = xbuf[0];
          xint[3] = xbuf[1];

          // vzint[0] = vz_ * delta[0];
          // vzint[1] = vz_ * (1-delta[0]);

        } else {
          // x and y cross
          xint[0] = 0.5 * dxi[0];
          delta[0]= ( xint[0] - x[part_idx+0*chunk_size] ) / ( xbuf[0] - x[part_idx+0*chunk_size]);
          xint[1] =  x[part_idx+1*chunk_size] + ( xbuf[1] - x[part_idx+1*chunk_size]) * delta[0];

          no_y_cross = xint[1] >= -0.5 && xint[1] < 0.5;
          i1 = !no_y_cross;
          i2 = no_y_cross;

          if (no_y_cross) {

            xint[3] = 0.5 * dxi[1];
            delta[1]= ( xint[3] - xint[1] ) / ( xbuf[1] - xint[1] );
            xint[2] = -xint[0] + ( xbuf[0] - xint[0] ) * delta[1];

            // vzint[0] = vz_ * delta[0];
            // vzint[1] = vz_ * delta[1] * (1-delta[0]);
            // vzint[2] = vz_ * (1-delta[0]) * (1-delta[1]);

          } else {

            xint[2] = xint[0]; xint[3] = xint[1];
            xint[1] = 0.5 * dxi[1];
            delta[1]= ( xint[1] - x[part_idx+1*chunk_size] ) / ( xint[3] - x[part_idx+1*chunk_size]);
            xint[0] = x[part_idx+0*chunk_size] + (xint[2] - x[part_idx+0*chunk_size]) * delta[1];
            xint[3] -= dxi[1];

            // vzint[0] = vz_ * delta[0] * delta[1];
            // vzint[1] = vz_ * delta[0] * (1-delta[1]);
            // vzint[2] = vz_ * (1-delta[0]);

          }

          vzint[2*no_y_cross] *= no_y_cross ? 1.0-delta[1] : delta[1];
          vzint[1] *= no_y_cross ? delta[1] : (1.0-delta[1]);

        }

        vzint[0] *= delta[0];
        vzint[1] *= no_y_cross ? 1.0-delta[0] : delta[0];
        vzint[2] *= 1.0-delta[0];

        ind = p_ind[0];
        x0[ind] = x[part_idx+0*chunk_size];
        y0[ind] = x[part_idx+1*chunk_size];
        x1[ind] = xint[0];
        y1[ind] = xint[1];
        qq[ind] = q[part_idx];
        vz[ind] = vzint[0];
        ii[ind] = ix[part_idx+0*chunk_size] - 1;
        jj[ind] = ix[part_idx+1*chunk_size] - 1;

        if (my_num_par>1) {

          posneg[i1] = -1;
          posneg[i2] =  1;
          dxi_fac_i[i1] = 1;
          dxi_fac_i[i2] = 0;
          dxi_fac_x[i1] = cross<3 ? 1 : 0;
          dxi_fac_x[i2] = 0;

          ind = p_ind[1];
          x0[ind] = posneg[0] * xint[0];
          y0[ind] = posneg[1] * xint[1];
          x1[ind] = xint[2] - dxi_fac_x[0]*dxi[0];
          y1[ind] = xint[3] - dxi_fac_x[1]*dxi[1];
          qq[ind] = q[part_idx];
          vz[ind] = vzint[1];
          ii[ind] = ix[part_idx+0*chunk_size] + dxi_fac_i[0]*dxi[0] - 1;
          jj[ind] = ix[part_idx+1*chunk_size] + dxi_fac_i[1]*dxi[1] - 1;

          if (my_num_par==3) {

            ind = p_ind[2];
            x0[ind] = posneg[1] * xint[2];
            y0[ind] = posneg[0] * xint[3];
            x1[ind] = xbuf[0] - dxi[0];
            y1[ind] = xbuf[1] - dxi[1];
            qq[ind] = q[part_idx];
            vz[ind] = vzint[2];
            ii[ind] = ix[part_idx+0*chunk_size] + dxi[0] - 1;
            jj[ind] = ix[part_idx+1*chunk_size] + dxi[1] - 1;

          }

        }

      }
      __syncthreads();

      // --------------

      // now accumulate jay looping through all virtual particles
      // and shifting grid indexes

      int nsplit = num_par_tot[0] + num_par_tot[1] + num_par_tot[2];
      int i = tx;

      while ( i < nsplit ) {

        // Normalize charge
        qnx = qq[i] * jnorm1;
        qny = qq[i] * jnorm2;
        qvz = c1_3 * qq[i] * vz[i];

        // get spline weights for x and y
        spline_s2( x0[i], S0x );
        spline_s2( x1[i], S1x );

        spline_s2( y0[i], S0y );
        spline_s2( y1[i], S1y );

        // get longitudinal motion weights
        // the last value is set to 0 so we can accumulate
        // all current components in a single pass
        wl_s2( qnx, x0[i], x1[i], wl1 );
        wl1[P1] = 0;

        wl_s2( qny, y0[i], y1[i], wl2 );
        wl2[P1] = 0;

        // get perpendicular motion weights
        // (a division by 2 was moved to the jnorm vars. above)
        for (int k1=P0; k1<=P1; ++k1){
          wp1[k1] = S0y[k1] + S1y[k1];
          wp2[k1] = S0x[k1] + S1x[k1];
        }

        // accumulate j1, j2, j3 in a single pass
        for (int k2=P0; k2<=P1; ++k2){
          for (int k1=P0; k1<=P1; ++k1){

            tmp1 = S0x[k1]*S0y[k2] + S1x[k1]*S1y[k2];
            tmp2 = S0x[k1]*S1y[k2] + S1x[k1]*S0y[k2];

            atomicAdd( jay_f2_til+0+3*(ii[i]+k1+(jj[i]+k2)*stride), wl1[k1] * wp1[k2] );
            atomicAdd( jay_f2_til+1+3*(ii[i]+k1+(jj[i]+k2)*stride), wp2[k1] * wl2[k2] );
            atomicAdd( jay_f2_til+2+3*(ii[i]+k1+(jj[i]+k2)*stride), qvz * ( tmp1 + 0.5 * tmp2 ) );

          }
        }

        i += BLOCK_SIZE;

      }

      // --------------
      if ( part_idx < num_par_proc_chunk ) {
        x[part_idx+0*chunk_size] = xbuf[0] - dxi[0];
        x[part_idx+1*chunk_size] = xbuf[1] - dxi[1];
        ix[part_idx+0*chunk_size] += dxi[0];
        ix[part_idx+1*chunk_size] += dxi[1];
      }

      // TODO note, this only works if the chunks size is divisible by the block size
      // see sort for how to do it more generally. Maybe we want to do that here too
      part_idx += BLOCK_SIZE;

    } // end loop over chunk of particles ``while ( part_idx < num_par_proc_chunk )``

    num_par_proc_tot -= num_par_proc_chunk;
    chunk++;

  } // end loop over chunks  ``while ( num_par_remainin > 0 )``

#undef NP

}

int main(void)
{
  // Set up number of tiles, doing 2D, linear interpolation
  int ntils = 2; // Number of tiles
  int ppc = 2; // Number of particles per cell in each direction
  int ncells = 6; // ncells x ncells cells in a tile
  int nchunk = 32; // Number of particles in one chunk
  int nblocks_requested = 16; // Number of blocks requested

  // Parameters calculated from above
  int nguard = 1; // Number of guard cells
  int lb = -nguard; int ub = ncells + nguard;
  int stride = ncells + 2*nguard;
  int tot_cells = 3 * stride * stride;
  int offset = 3 * ( stride + nguard );
  int npart = ncells * ncells * ppc * ppc;
  int nchunks = ( npart + nchunk - 1 ) / nchunk;
  double dxp = 1.0 / ppc;
  double dxp2 = dxp / 2.0;

  srand(1);

  // Initialize field and current arrays
  double *e[ntils], *b[ntils], *jay[ntils];
  for (int i = 0; i < ntils; i++) {
    // Allocate arrays for each tile
    e[i] = new double[tot_cells]; e[i] += offset;
    b[i] = new double[tot_cells]; b[i] += offset;
    jay[i] = new double[tot_cells]; jay[i] += offset;

    // Populate E and B with random numbers, zero out jay
    for (int k = lb; k < ub; k++) {
      for (int j = lb; j < ub; j++) {
        for (int l = 0; l < 3; l++) {
          e[i][l+3*(j+stride*k)] = ( ((double) rand()/RAND_MAX) - 0.5 ) * 2;
          b[i][l+3*(j+stride*k)] = ( ((double) rand()/RAND_MAX) - 0.5 ) * 2;
          // jay[i][j+stride*k] = 0.0;
        }
      }
    }
  }

  // Initialize particle arrays
  double **x[ntils], **p[ntils], **q[ntils];
  int **ix[ntils];
  for (int i=0; i < ntils; i++) {
    // Allocate pointer arrays to chunks
    x[i] = new double*[nchunks];
    p[i] = new double*[nchunks];
    q[i] = new double*[nchunks];
    ix[i] = new int*[nchunks];

    // Allocate chunks
    for (int j=0; j < nchunks; j++) {
      x[i][j] = new double[2*nchunk];
      p[i][j] = new double[3*nchunk];
      q[i][j] = new double[nchunk];
      ix[i][j] = new int[2*nchunk];
    }

    // Populate particle data
    for (int k=0; k < ncells; k++) {
      for (int j=0; j < ncells; j++) {
        for (int l=0; l < ppc*ppc; l++) {
          int id = l + ( j + k*ncells ) * ppc*ppc;
          int chunk_id = id / nchunk;
          int p_id = id - chunk_id * nchunk;
          int x2_id = l / ppc;
          int x1_id = l - x2_id * ppc;
          // x1, x2
          x[i][chunk_id][p_id] = -0.5 - dxp2 + (x1_id+1) * dxp;
          x[i][chunk_id][p_id+nchunk] = -0.5 - dxp2 + (x2_id+1) * dxp;
          // p1, p2, p3
          p[i][chunk_id][p_id         ] = ( ((double) rand()/RAND_MAX) - 0.5 ) * 2;
          p[i][chunk_id][p_id+  nchunk] = ( ((double) rand()/RAND_MAX) - 0.5 ) * 2;
          p[i][chunk_id][p_id+2*nchunk] = ( ((double) rand()/RAND_MAX) - 0.5 ) * 2;
          // q
          q[i][chunk_id][p_id] = 1.0;
          // ix1, ix2
          ix[i][chunk_id][p_id] = j;
          ix[i][chunk_id][p_id+nchunk] = k;
          // printf("%d, %d, %g, %g\n",ix[i][chunk_id][p_id],ix[i][chunk_id][p_id+nchunk],x[i][chunk_id][p_id],x[i][chunk_id][p_id+nchunk]);
        }
      }
    }
  }

  // Allocate the field/current arrays on the device

  // Host array of device pointers holding entire arrays
  double *d_e_arr[ntils], *d_b_arr[ntils], *d_jay_arr[ntils];
  // Device array of device pointers to first non-guard cell
  double **d_e, **d_b, **d_jay;
  // Host array of device pointers to first non-guard cell (to copy in)
  double *d_e_0[ntils], *d_b_0[ntils], *d_jay_0[ntils];

  for (int i = 0; i < ntils; i++) {
    // Allocate device arrays
    cudaMalloc( &d_e_arr[i], tot_cells*sizeof(double) );
    cudaMalloc( &d_b_arr[i], tot_cells*sizeof(double) );
    cudaMalloc( &d_jay_arr[i], tot_cells*sizeof(double) );

    // Copy in field data
    cudaMemcpy( d_e_arr[i], e[i]-offset, tot_cells*sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( d_b_arr[i], b[i]-offset, tot_cells*sizeof(double), cudaMemcpyHostToDevice );
    // cudaMemcpy( d_jay_arr[i], jay[i]-offset, tot_cells*sizeof(double), cudaMemcpyHostToDevice );

    d_e_0[i] = d_e_arr[i] + offset;
    d_b_0[i] = d_b_arr[i] + offset;
    d_jay_0[i] = d_jay_arr[i] + offset;
  }

  cudaMalloc( &d_e, ntils*sizeof(double*) );
  cudaMalloc( &d_b, ntils*sizeof(double*) );
  cudaMalloc( &d_jay, ntils*sizeof(double*) );

  cudaMemcpy( d_e, d_e_0, ntils*sizeof(double*), cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, d_b_0, ntils*sizeof(double*), cudaMemcpyHostToDevice );
  cudaMemcpy( d_jay, d_jay_0, ntils*sizeof(double*), cudaMemcpyHostToDevice );

  // Allocate the particle arrays on the device

  // Device array containing massive chunk list
  size_t x_offset = 0;
  size_t p_offset = x_offset + 2 * sizeof(double) * nchunk;
  size_t q_offset = p_offset + 3 * sizeof(double) * nchunk;
  size_t ix_offset = q_offset + sizeof(double) * nchunk;
  size_t chunk_mem = ix_offset + 2 * sizeof(int) * nchunk;
  char *d_chunks;
  cudaMalloc( &d_chunks, chunk_mem * nchunks * ntils );

  // Host array of device pointers to the beginning of each chunk
  double *d_x[ntils*nchunks], *d_p[ntils*nchunks], *d_q[ntils*nchunks];
  int *d_ix[ntils*nchunks];

  for (int i = 0; i < ntils; i++) {
    for (int j=0; j < nchunks; j++) {
      d_x[j+i*nchunks] = (double*) ( d_chunks + x_offset + (j+i*nchunks) * chunk_mem );
      d_p[j+i*nchunks] = (double*) ( d_chunks + p_offset + (j+i*nchunks) * chunk_mem );
      d_q[j+i*nchunks] = (double*) ( d_chunks + q_offset + (j+i*nchunks) * chunk_mem );
      d_ix[j+i*nchunks] = (int*) ( d_chunks + ix_offset + (j+i*nchunks) * chunk_mem );

      // Copy in particle data
      cudaMemcpy( d_x[j+i*nchunks], x[i][j], 2*nchunk*sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( d_p[j+i*nchunks], p[i][j], 3*nchunk*sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( d_q[j+i*nchunks], q[i][j], nchunk*sizeof(double), cudaMemcpyHostToDevice );
      cudaMemcpy( d_ix[j+i*nchunks], ix[i][j], 2*nchunk*sizeof(int), cudaMemcpyHostToDevice );
    }
  }

  // use one block per chunk if there aren't enough chunks (there should be
  // more chunks per blocks in ordinary usage)
  int nchunks_per_block = max( (nchunks*ntils) / nblocks_requested, 1 );
  int nblocks = max( nchunks / nchunks_per_block, 1 ) * ntils;

  DUDT_BORIS_2D<32> <<< nblocks, 32 >>>( d_chunks, nchunks_per_block, nchunks,
                                         ntils, npart, stride, -1.0, d_b, d_e, 0.1, nchunk );

  ADVANCE_DEPOSIT_2D<32> <<< nblocks, 32 >>>( d_chunks, nchunks_per_block, nchunks, ntils,
                                              npart, stride, 0.142, 0.142, d_jay, 0.1,
                                              nchunk );

  // Print old particle momentum
  for (int i=0; i < ntils; i++) {
    for (int k=0; k < ncells; k++) {
      for (int j=0; j < ncells; j++) {
        for (int l=0; l < ppc*ppc; l++) {
          int id = l + ( j + k*ncells ) * ppc*ppc;
          int chunk_id = id / nchunk;
          int p_id = id - chunk_id * nchunk;
          printf("%d, %d, %g, %g, %g, %g\n",ix[i][chunk_id][p_id],ix[i][chunk_id][p_id+nchunk],x[i][chunk_id][p_id],x[i][chunk_id][p_id+nchunk],p[i][chunk_id][p_id],p[i][chunk_id][p_id+nchunk]);
        }
      }
    }
  }

  printf("----------------\n");

  // Copy back particle data
  for (int i = 0; i < ntils; i++) {
    for (int j=0; j < nchunks; j++) {

      // Copy in particle data
      cudaMemcpy( x[i][j], d_x[j+i*nchunks], 2*nchunk*sizeof(double), cudaMemcpyDeviceToHost );
      cudaMemcpy( p[i][j], d_p[j+i*nchunks], 3*nchunk*sizeof(double), cudaMemcpyDeviceToHost );
      cudaMemcpy( ix[i][j], d_ix[j+i*nchunks], 2*nchunk*sizeof(int), cudaMemcpyDeviceToHost );
    }
  }

  // Print new particle momentum
  for (int i=0; i < ntils; i++) {
    for (int k=0; k < ncells; k++) {
      for (int j=0; j < ncells; j++) {
        for (int l=0; l < ppc*ppc; l++) {
          int id = l + ( j + k*ncells ) * ppc*ppc;
          int chunk_id = id / nchunk;
          int p_id = id - chunk_id * nchunk;
          printf("%d, %d, %g, %g, %g, %g\n",ix[i][chunk_id][p_id],ix[i][chunk_id][p_id+nchunk],x[i][chunk_id][p_id],x[i][chunk_id][p_id+nchunk],p[i][chunk_id][p_id],p[i][chunk_id][p_id+nchunk]);
        }
      }
    }
  }

  for (int i = 0; i < ntils; i++) {
    e[i] -= offset;
    b[i] -= offset;
    jay[i] -= offset;
    delete[] e[i];
    delete[] b[i];
    delete[] jay[i];
    for (int j=0; j < nchunks; j++) {
      delete[] x[i][j];
      delete[] p[i][j];
      delete[] q[i][j];
      delete[] ix[i][j];
    }
    delete[] x[i];
    delete[] p[i];
    delete[] q[i];
    delete[] ix[i];

    cudaFree( d_e_arr[i] );
    cudaFree( d_b_arr[i] );
    cudaFree( d_jay_arr[i] );
  }

  cudaFree( d_e );
  cudaFree( d_b );
  cudaFree( d_jay );

  cudaFree( d_chunks );
}