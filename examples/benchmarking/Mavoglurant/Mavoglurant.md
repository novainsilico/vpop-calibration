# Mavoglurant model:

## Parameters

6 parameters to optimise:

In the original model:
PDU: Clint
MI: KbBR,KbMU,KbBO,KbRB,KbAD

The original model appears to suffer from identifiability issues, with multiple parameter configurations that can explain the data equally well. We therefore conducted a parameter importance analysis in order to simplify the model

## Parameter importance

Considering all parameters as PDUs,

With nlmixr, we evaluate the Log-likelihood by varying one parameter at a tim (all other parameters being fixed to their original values):

KbMU: -937
KbBO: -866
KbRB: -853
KbBR: -883
KbAD: -1018
CLint: -592

This leads to the hypothesis that the most important parameters are, by decreasing order, CLint, KbRB, KbBO, KbBR, KbMU, KbAD.

If we optimise all parameters we have the following results:

Log-Lik: -158

── Population Parameters : ──

          Est.     SE  %RSE       Back-transformed(95%CI) BSV(CV%) Shrink(SD)%
lKbBR     2.25  0.834  37.1             9.47 (1.85, 48.6)     15.3      75.9%
lKbMU   -0.362  0.108  29.9          0.696 (0.563, 0.861)     69.8      18.6%
lKbAD     2.03 0.0419  2.06              7.64 (7.04, 8.3)     26.9      42.4%
lCLint    7.37 0.0392 0.532 1.58e+03 (1.47e+03, 1.71e+03)     43.4      3.06%
lKbBO      2.2  0.106  4.81             9.05 (7.35, 11.1)     127.      18.6%
lKbRB      1.1   1.36   123            3.01 (0.211, 43.2)     20.5      79.6%
add.err   0.21                                       0.21

The high standard error value of lKbRB (123%) as well as the shrinkage value (79.6%) leads to the belief that the data is not enough to identify KbRB correctly. Thus, there could be value in fixing KbBR to it's original value.
If we fix KbRB:

Log-Lik: -193


── Population Parameters : ──

         Est.     SE  %RSE       Back-transformed(95%CI) BSV(CV%) Shrink(SD)%
lKbBR    2.43  0.116  4.79             11.4 (9.04, 14.3)     13.6      77.5%
lKbMU   -0.28  0.101  36.1           0.756 (0.62, 0.921)     63.8      20.6%
lKbAD    2.05 0.0416  2.03             7.79 (7.18, 8.45)     26.5      44.1%
lCLint   7.37 0.0391 0.531 1.58e+03 (1.46e+03, 1.71e+03)     43.2      3.06%
lKbBO    2.18  0.104  4.76             8.86 (7.23, 10.9)     122.      19.0%
add.err 0.217                                      0.217

The Log-Likelihood has not decreased dramatically et no parameter has a high standard error. However KbBR has a high shrinkage value and we could envisage fixing it.
If we fix KbMU:

Log-Lik: -337

── Population Parameters : ──

          Est.     SE  %RSE       Back-transformed(95%CI) BSV(CV%) Shrink(SD)%
lKbMU   0.0995 0.0418    42               1.1 (1.02, 1.2)     31.4      32.3%
lKbAD     2.09 0.0423  2.02             8.06 (7.42, 8.76)     26.4      43.8%
lCLint    7.35 0.0383 0.521 1.56e+03 (1.44e+03, 1.68e+03)     41.8      3.76%
lKbBO     2.11 0.0973   4.6               8.29 (6.85, 10)     103.      21.9%
add.err  0.247                                      0.247

Now no parameters present a high standard error value or shrinkage value. However the Log-Likelihood has decreased indicating a degradation of the model's ability to correctly explain the data.
