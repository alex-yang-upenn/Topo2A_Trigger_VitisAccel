//Numpy array shape [32]
//Min -0.187500000000
//Max 0.187500000000
//Number of zeros 12

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[32];
#else
bias2_t b2[32] = {0.0000, -0.1875, -0.0625, 0.0000, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, 0.0000, -0.0625, -0.0625, -0.0625, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1875, -0.1875, 0.0625, 0.0625, -0.0625, -0.0625, 0.1250, 0.0000, -0.0625, 0.0000, 0.0000};
#endif

#endif
