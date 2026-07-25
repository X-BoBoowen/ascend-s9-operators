#include <stdint.h>
/**
 * @brief origin source from freebsd-src
 * 
 */



/*
 * A union which permits us to convert between a float and a 32 bit
 * int.
 */

typedef union
{
  float value;
  /* FIXME: Assumes 32 bit int.  */
  unsigned int word;
} ieee_float_shape_type;

/* Get a 32 bit int from a float.  */

#define GET_FLOAT_WORD(i,d)					\
do {								\
  ieee_float_shape_type gf_u;					\
  gf_u.value = (d);						\
  (i) = gf_u.word;						\
} while (0)

/* Set a float from a 32 bit int.  */

#define SET_FLOAT_WORD(d,i)					\
do {								\
  ieee_float_shape_type sf_u;					\
  sf_u.word = (i);						\
  (d) = sf_u.value;						\
} while (0)

// #ifdef FLT_EVAL_METHOD
/*
 * Attempt to get strict C99 semantics for assignment with non-C99 compilers.
 */
#if FLT_EVAL_METHOD == 0 || __GNUC__ == 0
#define	STRICT_ASSIGN(type, lval, rval)	((lval) = (rval))
#else
#define	STRICT_ASSIGN(type, lval, rval) do {	\
	volatile type __lval;			\
						\
	if (sizeof(type) >= sizeof(long double))	\
		(lval) = (rval);		\
	else {					\
		__lval = (rval);		\
		(lval) = __lval;		\
	}					\
} while (0)
#endif
// #endif /* FLT_EVAL_METHOD */

__aicore__ inline
float
mylog1pf(float x)
{
    static const float
ln2_hi =   6.9313812256e-01,	/* 0x3f317180 */
ln2_lo =   9.0580006145e-06,	/* 0x3717f7d1 */
two25 =    3.355443200e+07,	/* 0x4c000000 */
Lp1 = 6.6666668653e-01,	/* 3F2AAAAB */
Lp2 = 4.0000000596e-01,	/* 3ECCCCCD */
Lp3 = 2.8571429849e-01, /* 3E924925 */
Lp4 = 2.2222198546e-01, /* 3E638E29 */
Lp5 = 1.8183572590e-01, /* 3E3A3325 */
Lp6 = 1.5313838422e-01, /* 3E1CD04F */
Lp7 = 1.4798198640e-01; /* 3E178897 */

static const float zero = 0.0;
static volatile float vzero = 0.0;
    
	float hfsq,f,c,s,z,R,u;
	int32_t k,hx,hu,ax;

	GET_FLOAT_WORD(hx,x);
	ax = hx&0x7fffffff;

	k = 1;
	if (hx < 0x3ed413d0) {			/* 1+x < sqrt(2)+  */
	    if(ax>=0x3f800000) {		/* x <= -1.0 */
		if(x==(float)-1.0) return -two25/vzero; /* log1p(-1)=+inf */
		else return (x-x)/(x-x);	/* log1p(x<-1)=NaN */
	    }
	    if(ax<0x38000000) {			/* |x| < 2**-15 */
		if(two25+x>zero			/* raise inexact */
	            &&ax<0x33800000) 		/* |x| < 2**-24 */
		    return x;
		else
		    return x - x*x*(float)0.5;
	    }
	    if(hx>0||hx<=((int32_t)0xbe95f619)) {
		k=0;f=x;hu=1;}		/* sqrt(2)/2- <= 1+x < sqrt(2)+ */
	}
	if (hx >= 0x7f800000) return x+x;
	if(k!=0) {
	    if(hx<0x5a000000) {
		STRICT_ASSIGN(float,u,(float)1.0+x);
		GET_FLOAT_WORD(hu,u);
	        k  = (hu>>23)-127;
		/* correction term */
	        c  = (k>0)? (float)1.0-(u-x):x-(u-(float)1.0);
		c /= u;
	    } else {
		u  = x;
		GET_FLOAT_WORD(hu,u);
	        k  = (hu>>23)-127;
		c  = 0;
	    }
	    hu &= 0x007fffff;
	    /*
	     * The approximation to sqrt(2) used in thresholds is not
	     * critical.  However, the ones used above must give less
	     * strict bounds than the one here so that the k==0 case is
	     * never reached from here, since here we have committed to
	     * using the correction term but don't use it if k==0.
	     */
	    if(hu<0x3504f4) {			/* u < sqrt(2) */
	        SET_FLOAT_WORD(u,hu|0x3f800000);/* normalize u */
	    } else {
	        k += 1;
		SET_FLOAT_WORD(u,hu|0x3f000000);	/* normalize u/2 */
	        hu = (0x00800000-hu)>>2;
	    }
	    f = u-(float)1.0;
	}
	hfsq=(float)0.5*f*f;
	if(hu==0) {	/* |f| < 2**-20 */
	    if(f==zero) {
		if(k==0) {
		    return zero;
		} else {
		    c += k*ln2_lo;
		    return k*ln2_hi+c;
		}
	    }
	    R = hfsq*((float)1.0-(float)0.66666666666666666*f);
	    if(k==0) return f-R; else
	    	     return k*ln2_hi-((R-(k*ln2_lo+c))-f);
	}
 	s = f/((float)2.0+f);
	z = s*s;
	R = z*(Lp1+z*(Lp2+z*(Lp3+z*(Lp4+z*(Lp5+z*(Lp6+z*Lp7))))));
	if(k==0) return f-(hfsq-s*(hfsq+R)); else
		 return k*ln2_hi-((hfsq-(s*(hfsq+R)+(k*ln2_lo+c)))-f);
}

//--------------------------------------------


#define u_int32_t uint32_t

__aicore__ inline float
myexpf(float x)
{
    static const float
one	= 1.0,
halF[2]	= {0.5,-0.5,},
o_threshold=  8.8721679688e+01,  /* 0x42b17180 */
u_threshold= -1.0397208405e+02,  /* 0xc2cff1b5 */
ln2HI[2]   ={ 6.9314575195e-01,		/* 0x3f317200 */
	     -6.9314575195e-01,},	/* 0xbf317200 */
ln2LO[2]   ={ 1.4286067653e-06,  	/* 0x35bfbe8e */
	     -1.4286067653e-06,},	/* 0xb5bfbe8e */
invln2 =  1.4426950216e+00, 		/* 0x3fb8aa3b */
/*
 * Domain [-0.34568, 0.34568], range ~[-4.278e-9, 4.447e-9]:
 * |x*(exp(x)+1)/(exp(x)-1) - p(x)| < 2**-27.74
 */
P1 =  1.6666625440e-1,		/*  0xaaaa8f.0p-26 */
P2 = -2.7667332906e-3;		/* -0xb55215.0p-32 */

static volatile float
huge	= 1.0e+30,
twom100 = 7.8886090522e-31;      /* 2**-100=0x0d800000 */

	float y,hi=0.0,lo=0.0,c,t,twopk;
	int32_t k=0,xsb;
	u_int32_t hx;

	GET_FLOAT_WORD(hx,x);
	xsb = (hx>>31)&1;		/* sign bit of x */
	hx &= 0x7fffffff;		/* high word of |x| */

    /* filter out non-finite argument */
	if(hx >= 0x42b17218) {			/* if |x|>=88.721... */
	    if(hx>0x7f800000)
		 return x+x;	 		/* NaN */
            if(hx==0x7f800000)
		return (xsb==0)? x:0.0f;		/* exp(+-inf)={inf,0} */
	    if(x > o_threshold) return huge*huge; /* overflow */
	    if(x < u_threshold) return twom100*twom100; /* underflow */
	}

    /* argument reduction */
	if(hx > 0x3eb17218) {		/* if  |x| > 0.5 ln2 */
	    if(hx < 0x3F851592) {	/* and |x| < 1.5 ln2 */
		hi = x-ln2HI[xsb]; lo=ln2LO[xsb]; k = 1-xsb-xsb;
	    } else {
		k  = invln2*x+halF[xsb];
		t  = k;
		hi = x - t*ln2HI[0];	/* t*ln2HI is exact here */
		lo = t*ln2LO[0];
	    }
	    STRICT_ASSIGN(float, x, hi - lo);
	}
	else if(hx < 0x39000000)  {	/* when |x|<2**-14 */
	    if(huge+x>one) return one+x;/* trigger inexact */
	}
	else k = 0;

    /* x is now in primary range */
	t  = x*x;
	if(k >= -125)
	    SET_FLOAT_WORD(twopk,((u_int32_t)(0x7f+k))<<23);
	else
	    SET_FLOAT_WORD(twopk,((u_int32_t)(0x7f+(k+100)))<<23);
	c  = x - t*(P1+t*P2);
	if(k==0) 	return one-((x*c)/(c-(float)2.0)-x);
	else 		y = one-((lo-(x*c)/((float)2.0-c))-hi);
	if(k >= -125) {
	    if(k==128) return y*2.0F*0x1p127F;
	    return y*twopk;
	} else {
	    return y*twopk*twom100;
	}
}
