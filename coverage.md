| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| vpop\_calibration/api/\_\_init\_\_.py                  |        8 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/api/interface.py                     |       53 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/compatibility.py                     |       14 |       12 |        0 |        0 |     14% |      3-19 |
| vpop\_calibration/config.py                            |        9 |        1 |        2 |        1 |     82% |         7 |
| vpop\_calibration/data\_generation.py                  |       91 |        2 |       12 |        4 |     94% |59, 73, 128-\>131, 167-\>170 |
| vpop\_calibration/metropolis\_hastings.py              |       54 |        1 |        8 |        1 |     97% |       133 |
| vpop\_calibration/model/\_\_init\_\_.py                |        2 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/model/data.py                        |      162 |       23 |       44 |       14 |     80% |51, 53, 55, 57, 170, 235, 283-293, 367, 374, 381, 386, 390, 394, 401-403, 452 |
| vpop\_calibration/model/gp.py                          |      227 |       49 |       68 |       20 |     75% |61, 66, 102, 139, 141-142, 146-149, 224-225, 232, 252, 286, 308, 310, 364, 385, 389-394, 401-409, 411-\>346, 435-437, 465, 475-484, 514-\>exit, 589-596 |
| vpop\_calibration/model/plot.py                        |      122 |       11 |       32 |        8 |     88% |76-77, 147-148, 174-\>177, 219-\>178, 224-225, 244, 253-\>255, 256-259 |
| vpop\_calibration/pynlme/\_\_init\_\_.py               |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/conditional\_distribution.py  |      216 |       31 |       68 |       11 |     79% |9-10, 13-14, 145, 150-\>152, 157-\>exit, 162-164, 234-\>239, 249, 264-274, 277-278, 299-306, 309-315, 387 |
| vpop\_calibration/pynlme/config.py                     |       17 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/data.py                       |       61 |        0 |       12 |        0 |    100% |           |
| vpop\_calibration/pynlme/diagnostics.py                |      169 |       11 |       24 |        7 |     90% |88-89, 161-\>164, 209-211, 230-\>234, 308, 328, 399, 409-415 |
| vpop\_calibration/pynlme/error\_estimation.py          |       38 |        0 |       10 |        0 |    100% |           |
| vpop\_calibration/pynlme/fim/\_\_init\_\_.py           |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/fim/display.py                |       26 |        9 |       10 |        2 |     53% |5-6, 15-21, 26-\>28, 35-\>37 |
| vpop\_calibration/pynlme/fim/estimator.py              |       84 |       12 |       22 |       11 |     78% |60, 67, 77, 86, 90, 96, 102, 113, 127, 135, 144, 152 |
| vpop\_calibration/pynlme/fim/likelihood\_derivation.py |       64 |        0 |       14 |        0 |    100% |           |
| vpop\_calibration/pynlme/fim/standard\_error.py        |       14 |        1 |        2 |        1 |     88% |         8 |
| vpop\_calibration/pynlme/fim/state.py                  |       68 |        2 |        6 |        1 |     96% |     87-88 |
| vpop\_calibration/pynlme/fim/utils.py                  |       27 |        6 |        2 |        0 |     72% |     23-29 |
| vpop\_calibration/pynlme/importance\_sampling.py       |       56 |        0 |       10 |        1 |     98% |   85-\>88 |
| vpop\_calibration/pynlme/indexing.py                   |       66 |        0 |        8 |        0 |    100% |           |
| vpop\_calibration/pynlme/initial\_estimates.py         |       78 |       60 |       22 |        0 |     18% |15-37, 45-70, 73-76, 82-100, 113-123, 126-131, 134-140 |
| vpop\_calibration/pynlme/model.py                      |      305 |        6 |       34 |        6 |     96% |89-\>96, 213-215, 262-\>258, 303, 497, 745-\>751, 754 |
| vpop\_calibration/pynlme/params.py                     |      171 |        4 |       42 |        4 |     96% |54, 56, 126-\>129, 128, 208 |
| vpop\_calibration/pynlme/plot.py                       |      435 |       45 |      134 |       37 |     86% |39-\>41, 73, 75-76, 122-123, 134, 159, 192-193, 205-\>209, 210, 219-220, 245, 279-280, 295, 299-\>305, 311, 316, 354, 382-383, 396, 438, 440, 453-454, 499, 513, 519, 525, 529-530, 538, 656, 665, 667-\>670, 697-\>708, 745-746, 758-759, 844-845, 855, 862, 926-927 |
| vpop\_calibration/pynlme/residuals.py                  |       96 |        1 |        6 |        1 |     98% |        42 |
| vpop\_calibration/pynlme/schemas.py                    |       25 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/pynlme/utils.py                      |       26 |        4 |        2 |        0 |     79% |     82-85 |
| vpop\_calibration/saem/\_\_init\_\_.py                 |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/config.py                       |       29 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/estimates.py                    |       97 |       17 |       26 |        0 |     78% |165-174, 177-184, 204, 208 |
| vpop\_calibration/saem/fixed\_effects.py               |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/saem/m\_step.py                      |       54 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/saem/optimizer.py                    |      143 |       18 |       42 |        8 |     82% |37, 42-\>54, 155, 162-165, 168-171, 174-181, 277-\>307, 384-385, 401-409 |
| vpop\_calibration/saem/plot.py                         |       47 |       36 |       14 |        0 |     18% |3-4, 7-8, 24-57, 60-68, 71 |
| vpop\_calibration/saem/scheduler.py                    |       48 |        0 |       14 |        0 |    100% |           |
| vpop\_calibration/saem/utils.py                        |       23 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/\_\_init\_\_.py                  |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/config.py                        |        7 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/diagnostics.py                   |       43 |        4 |       12 |        6 |     82% |47-48, 52, 60, 62-\>65, 65-\>68, 68-\>72 |
| vpop\_calibration/sdk/model.py                         |       46 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/saem.py                          |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/\_\_init\_\_.py    |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/analytical.py      |       41 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/base.py            |       30 |        1 |        0 |        0 |     97% |        38 |
| vpop\_calibration/structural\_model/gp.py              |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/sbml.py            |       90 |        8 |       12 |        2 |     88% |29-30, 61-66, 69 |
| vpop\_calibration/structural\_model/simwork.py         |      128 |        7 |       22 |        2 |     91% |72-89, 121 |
| vpop\_calibration/utils.py                             |       28 |        3 |        6 |        2 |     85% | 29-30, 35 |
| **TOTAL**                                              | **3700** |  **385** |  **752** |  **150** | **86%** |           |
