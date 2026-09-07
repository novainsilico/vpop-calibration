| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| vpop\_calibration/api/\_\_init\_\_.py                 |        8 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/api/interface.py                    |       53 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/compatibility.py                    |       14 |       12 |        0 |        0 |     14% |      3-19 |
| vpop\_calibration/config.py                           |        9 |        1 |        2 |        1 |     82% |         7 |
| vpop\_calibration/data\_generation.py                 |       91 |        2 |       12 |        4 |     94% |59, 73, 128-\>131, 167-\>170 |
| vpop\_calibration/metropolis\_hastings.py             |       54 |        1 |        8 |        1 |     97% |       133 |
| vpop\_calibration/model/\_\_init\_\_.py               |        2 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/model/data.py                       |      162 |       23 |       44 |       14 |     80% |51, 53, 55, 57, 170, 235, 283-293, 367, 374, 381, 386, 390, 394, 401-403, 452 |
| vpop\_calibration/model/gp.py                         |      227 |       49 |       68 |       20 |     75% |61, 66, 102, 139, 141-142, 146-149, 224-225, 232, 252, 286, 308, 310, 364, 385, 389-394, 401-409, 411-\>346, 435-437, 465, 475-484, 514-\>exit, 589-596 |
| vpop\_calibration/model/plot.py                       |      122 |       11 |       32 |        8 |     88% |76-77, 147-148, 174-\>177, 219-\>178, 224-225, 244, 253-\>255, 256-259 |
| vpop\_calibration/pynlme/\_\_init\_\_.py              |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/conditional\_distribution.py |      185 |       18 |       44 |        5 |     86% |9-10, 13-14, 131, 136-\>138, 144-148, 236, 257-263, 335 |
| vpop\_calibration/pynlme/config.py                    |       14 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/data.py                      |       61 |        0 |       12 |        0 |    100% |           |
| vpop\_calibration/pynlme/diagnostics.py               |      164 |        8 |       22 |        7 |     92% |84-85, 157-\>160, 205-207, 226-\>230, 304, 324, 395 |
| vpop\_calibration/pynlme/error\_estimation.py         |       38 |        0 |       10 |        0 |    100% |           |
| vpop\_calibration/pynlme/importance\_sampling.py      |       56 |        0 |       10 |        1 |     98% |   85-\>88 |
| vpop\_calibration/pynlme/indexing.py                  |       66 |        0 |        8 |        0 |    100% |           |
| vpop\_calibration/pynlme/initial\_estimates.py        |       78 |       60 |       22 |        0 |     18% |15-37, 45-70, 73-76, 82-100, 113-123, 126-131, 134-140 |
| vpop\_calibration/pynlme/model.py                     |      267 |       21 |       28 |        3 |     92% |83-\>90, 242-\>238, 463, 646-691 |
| vpop\_calibration/pynlme/params.py                    |      180 |        4 |       48 |        4 |     96% |54, 56, 126-\>129, 128, 226 |
| vpop\_calibration/pynlme/plot.py                      |      435 |       45 |      134 |       37 |     86% |39-\>41, 73, 75-76, 122-123, 134, 159, 192-193, 205-\>209, 210, 219-220, 245, 279-280, 295, 299-\>305, 311, 316, 354, 382-383, 396, 438, 440, 453-454, 499, 513, 519, 525, 529-530, 538, 656, 665, 667-\>670, 697-\>708, 745-746, 758-759, 844-845, 855, 862, 926-927 |
| vpop\_calibration/pynlme/residuals.py                 |       91 |        1 |        4 |        1 |     98% |        42 |
| vpop\_calibration/pynlme/schemas.py                   |       25 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/pynlme/utils.py                     |       26 |        4 |        2 |        0 |     79% |     82-85 |
| vpop\_calibration/saem/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/config.py                      |       29 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/estimates.py                   |       97 |       17 |       26 |        0 |     78% |158-167, 170-177, 197, 201 |
| vpop\_calibration/saem/fixed\_effects.py              |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/saem/m\_step.py                     |       53 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/saem/optimizer.py                   |      147 |       19 |       46 |        9 |     81% |38, 43-\>55, 156, 167-170, 173-176, 179-186, 188-\>exit, 281-\>311, 388-389, 405-414 |
| vpop\_calibration/saem/plot.py                        |       50 |       39 |       20 |        0 |     16% |3-4, 7-8, 24-59, 62-70, 73-74 |
| vpop\_calibration/saem/scheduler.py                   |       48 |        0 |       14 |        0 |    100% |           |
| vpop\_calibration/saem/utils.py                       |       27 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/\_\_init\_\_.py                 |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/config.py                       |        7 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/diagnostics.py                  |       43 |        4 |       12 |        6 |     82% |47-48, 52, 60, 62-\>65, 65-\>68, 68-\>72 |
| vpop\_calibration/sdk/model.py                        |       46 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/saem.py                         |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/\_\_init\_\_.py   |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/analytical.py     |       41 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/base.py           |       30 |        1 |        0 |        0 |     97% |        38 |
| vpop\_calibration/structural\_model/gp.py             |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/sbml.py           |       90 |        8 |       12 |        2 |     88% |29-30, 61-66, 69 |
| vpop\_calibration/structural\_model/simwork.py        |      128 |        7 |       22 |        2 |     91% |72-89, 121 |
| vpop\_calibration/utils.py                            |       24 |        3 |        6 |        2 |     83% | 29-30, 35 |
| **TOTAL**                                             | **3350** |  **358** |  **678** |  **127** | **86%** |           |
