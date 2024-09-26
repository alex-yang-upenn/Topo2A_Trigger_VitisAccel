//Numpy array shape [32, 16]
//Min -0.625000000000
//Max 0.500000000000
//Number of zeros 65

#ifndef W6_H_
#define W6_H_

#ifndef __SYNTHESIS__
weight6_t w6[512];
#else
weight6_t w6[512] = {0.0625, 0.3125, 0.3125, -0.1250, 0.5000, -0.1875, 0.1875, 0.0625, -0.1250, 0.2500, -0.2500, -0.0625, 0.3750, 0.3125, -0.5000, -0.0625, -0.1250, 0.0625, 0.4375, 0.0625, 0.3125, -0.3125, -0.4375, -0.1875, 0.1250, -0.0625, -0.2500, 0.3125, -0.1250, 0.2500, 0.0625, -0.0625, 0.1875, -0.0625, 0.0625, -0.1875, -0.3125, 0.3125, -0.1875, 0.0000, 0.0000, -0.0625, 0.1875, -0.1875, -0.0625, 0.0625, 0.3125, -0.3750, -0.3125, 0.1250, 0.1250, -0.2500, 0.1250, -0.1250, -0.1250, 0.0000, 0.3125, 0.1875, 0.1250, -0.1250, 0.0000, 0.1875, -0.5000, -0.1250, 0.2500, 0.2500, -0.2500, 0.3750, 0.1875, -0.1250, -0.3750, 0.1250, -0.0625, 0.3750, -0.1875, 0.0625, 0.0000, -0.0625, -0.1250, 0.0625, 0.0000, 0.0000, -0.6250, -0.4375, -0.1250, 0.2500, -0.3125, 0.2500, 0.3125, 0.0000, 0.3750, -0.5625, 0.3125, -0.1875, 0.0625, -0.1250, -0.1250, 0.0000, 0.1875, -0.0625, -0.1250, -0.0625, 0.0000, -0.2500, -0.0625, -0.2500, 0.4375, -0.5625, 0.1250, -0.0625, 0.1250, -0.1875, 0.3750, 0.2500, 0.1875, 0.1250, 0.1250, 0.0625, 0.0625, -0.4375, -0.1875, 0.0625, -0.1875, 0.3125, 0.0000, 0.1250, -0.1875, 0.0000, 0.1250, 0.3125, 0.3125, 0.5000, -0.1875, -0.2500, 0.1875, -0.1875, 0.1250, 0.4375, -0.1875, -0.1250, -0.1250, 0.1875, -0.3125, 0.0625, -0.1875, -0.0625, 0.0000, -0.1250, 0.0625, 0.1875, -0.1250, 0.1250, -0.4375, -0.0625, 0.1250, 0.0625, 0.0000, 0.1250, 0.1250, -0.5625, 0.1250, 0.0000, 0.2500, 0.2500, 0.0000, -0.1250, -0.1875, 0.1250, 0.0625, 0.0625, 0.0000, 0.1875, 0.1250, -0.1250, 0.1250, -0.2500, -0.0625, 0.1250, 0.0000, 0.1875, 0.1875, 0.1250, -0.3125, -0.2500, 0.3750, 0.0625, 0.0000, 0.3750, -0.1875, -0.1250, -0.1250, -0.4375, 0.0000, 0.0625, 0.1875, 0.3125, 0.2500, 0.1875, 0.0000, -0.3750, 0.0000, -0.3750, 0.1250, -0.1250, -0.3125, -0.0625, 0.2500, -0.2500, 0.0000, 0.3125, 0.1250, 0.2500, -0.0625, -0.3125, -0.4375, 0.1875, 0.1250, 0.4375, 0.2500, -0.5000, 0.0000, 0.0000, 0.1875, 0.0000, -0.2500, 0.2500, -0.1875, -0.1875, -0.0625, -0.5625, -0.3750, -0.1250, -0.1250, 0.1875, 0.3125, -0.2500, -0.1875, 0.2500, 0.1250, 0.1875, 0.1875, 0.1875, -0.2500, 0.3125, 0.1875, -0.1875, -0.2500, 0.0000, 0.2500, -0.1250, 0.0000, -0.2500, -0.3125, 0.0000, 0.1875, 0.0000, -0.2500, -0.1875, -0.2500, -0.1875, -0.0625, -0.3750, -0.2500, -0.3750, -0.4375, -0.4375, -0.1250, 0.0000, -0.1250, -0.1250, 0.3750, -0.1250, 0.1250, 0.0000, 0.1875, 0.3125, -0.1875, -0.3750, -0.1250, 0.3125, -0.4375, 0.1875, 0.0000, -0.1250, -0.0625, 0.1875, 0.0625, 0.0000, 0.3125, 0.3125, -0.3125, 0.1250, 0.0000, -0.5000, -0.2500, -0.2500, -0.0625, 0.0000, 0.0000, -0.2500, -0.0625, 0.1250, 0.1250, -0.0625, -0.0625, -0.0625, 0.0000, -0.1875, -0.0625, 0.0000, -0.2500, 0.1250, -0.4375, -0.2500, 0.4375, 0.0625, 0.0625, 0.1875, 0.0000, -0.0625, 0.1250, 0.1250, 0.0625, 0.0000, -0.1875, -0.3750, 0.3750, -0.0625, -0.5000, 0.2500, -0.0625, 0.0625, 0.0625, 0.1875, 0.0000, -0.2500, 0.0000, 0.0000, -0.1875, -0.0625, -0.1250, -0.2500, 0.1875, -0.4375, -0.1250, -0.0625, 0.1250, -0.3125, 0.2500, -0.0625, -0.2500, -0.0625, -0.1875, -0.2500, -0.1250, -0.4375, 0.3125, -0.3125, -0.1250, 0.3750, 0.3125, 0.0625, -0.4375, 0.1875, -0.0625, -0.1250, -0.0625, 0.2500, -0.2500, 0.0000, -0.2500, -0.3125, 0.0000, -0.0625, -0.4375, 0.2500, -0.3125, 0.0625, 0.0000, -0.1875, -0.3125, 0.0000, -0.0625, -0.1250, 0.0000, -0.1250, -0.3125, -0.1875, 0.0625, 0.1250, 0.0000, -0.0625, 0.0625, 0.1875, 0.2500, -0.5625, -0.1250, -0.1250, -0.3750, -0.0625, 0.0000, 0.2500, -0.3750, 0.0000, 0.5000, 0.3750, -0.2500, -0.0625, 0.0625, 0.2500, -0.5000, -0.1250, -0.1875, 0.0000, 0.1250, 0.3125, 0.0000, 0.0625, 0.0000, 0.1250, 0.4375, -0.1250, -0.1250, -0.0625, 0.1250, 0.0625, 0.0000, 0.0000, -0.3750, -0.0625, 0.3750, -0.0625, 0.0625, -0.1875, -0.1250, -0.3750, -0.1250, 0.0625, -0.1250, 0.2500, -0.2500, -0.0625, -0.1875, -0.3125, 0.1250, 0.0625, -0.1250, -0.5625, -0.3125, 0.1250, 0.0000, -0.1875, 0.2500, 0.0000, -0.0625, -0.4375, 0.3125, 0.1875, 0.5000, -0.1250, -0.1250, 0.1875, 0.0000, -0.1875, 0.2500, 0.2500, 0.0625, -0.0625, 0.1250, -0.3750, 0.1875, 0.3125, -0.0625, 0.3750, 0.0625, 0.0000, -0.0625, 0.3125, 0.3125, 0.2500, -0.3125, -0.3125, 0.3750, 0.0000, -0.1250, -0.0625, 0.2500, 0.1875, 0.0000, 0.0625, -0.5000, 0.0625, -0.1250, 0.0000, -0.1250, -0.1250, -0.1250, -0.1250, 0.0625, 0.0000, -0.0625, -0.3125, -0.1875, -0.3750, 0.4375, -0.3125, -0.5000, -0.4375, 0.2500, -0.1875, -0.0625, 0.2500};
#endif

#endif
