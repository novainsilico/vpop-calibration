| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| vpop\_calibration/api/\_\_init\_\_.py                 |        8 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/api/interface.py                    |       53 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/compatibility.py                    |       14 |       12 |        0 |        0 |     14% |      3-19 |
| vpop\_calibration/config.py                           |        9 |        1 |        2 |        1 |     82% |         7 |
| vpop\_calibration/data\_generation.py                 |       91 |        2 |       12 |        4 |     94% |59, 73, 128-\>131, 167-\>170 |
| vpop\_calibration/metropolis\_hastings.py             |       59 |        1 |        8 |        1 |     97% |       137 |
| vpop\_calibration/model/\_\_init\_\_.py               |        2 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/model/data.py                       |      162 |       23 |       44 |       14 |     80% |51, 53, 55, 57, 171, 236, 284-294, 368, 375, 382, 387, 391, 395, 402-404, 453 |
| vpop\_calibration/model/gp.py                         |      227 |       49 |       68 |       20 |     75% |61, 66, 102, 139, 141-142, 146-149, 224-225, 232, 252, 286, 308, 310, 364, 385, 389-394, 401-409, 411-\>346, 435-437, 465, 475-484, 514-\>exit, 589-596 |
| vpop\_calibration/model/plot.py                       |      122 |       11 |       32 |        8 |     88% |76-77, 147-148, 174-\>177, 219-\>178, 224-225, 244, 253-\>255, 256-259 |
| vpop\_calibration/pynlme/\_\_init\_\_.py              |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/conditional\_distribution.py |      187 |       26 |       44 |        5 |     82% |9-10, 13-14, 132, 137-\>139, 145-149, 237, 258-264, 309-327, 346 |
| vpop\_calibration/pynlme/config.py                    |       14 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/data.py                      |       61 |        0 |       12 |        0 |    100% |           |
| vpop\_calibration/pynlme/diagnostics.py               |      176 |       11 |       28 |        9 |     90% |87-88, 159-\>162, 204, 218-219, 254, 256, 290, 302, 374, 478 |
| vpop\_calibration/pynlme/error\_estimation.py         |       40 |        0 |       12 |        0 |    100% |           |
| vpop\_calibration/pynlme/importance\_sampling.py      |       64 |        0 |       12 |        1 |     99% |   87-\>90 |
| vpop\_calibration/pynlme/indexing.py                  |       66 |        0 |        8 |        0 |    100% |           |
| vpop\_calibration/pynlme/initial\_estimates.py        |       79 |       61 |       22 |        0 |     18% |15-37, 45-72, 75-78, 84-102, 115-125, 128-133, 136-142 |
| vpop\_calibration/pynlme/model.py                     |      250 |        1 |       28 |        3 |     99% |83-\>90, 242-\>238, 463 |
| vpop\_calibration/pynlme/params.py                    |      180 |        4 |       48 |        4 |     96% |57, 59, 133-\>144, 141, 241 |
| vpop\_calibration/pynlme/plot.py                      |      429 |       44 |      130 |       36 |     86% |39-\>41, 73, 75-76, 122-123, 134, 159, 192-193, 205-\>209, 210, 219-220, 245, 279-280, 295, 299-\>305, 311, 316, 354, 382-383, 396, 438, 440, 453-454, 499, 513, 521, 525-526, 534, 654, 663, 665-\>668, 695-\>706, 743-744, 756-757, 842-843, 853, 860, 924-925 |
| vpop\_calibration/pynlme/residuals.py                 |      102 |        1 |       10 |        1 |     98% |        51 |
| vpop\_calibration/pynlme/schemas.py                   |       25 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/pynlme/utils.py                     |       26 |        4 |        2 |        0 |     79% |     82-85 |
| vpop\_calibration/saem/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/config.py                      |       29 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/estimates.py                   |      103 |       21 |       30 |        1 |     74% |161-166, 181-196, 199-206, 226, 230 |
| vpop\_calibration/saem/fixed\_effects.py              |       32 |        1 |        4 |        1 |     94% |        54 |
| vpop\_calibration/saem/m\_step.py                     |       53 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/saem/optimizer.py                   |      165 |       18 |       50 |        6 |     85% |158, 179-182, 185-188, 191-198, 200-\>exit, 436-437, 453-462 |
| vpop\_calibration/saem/plot.py                        |       50 |       39 |       20 |        0 |     16% |3-4, 7-8, 24-59, 62-70, 73-74 |
| vpop\_calibration/saem/scheduler.py                   |       48 |        0 |       14 |        0 |    100% |           |
| vpop\_calibration/saem/utils.py                       |       27 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/\_\_init\_\_.py                 |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/config.py                       |        7 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/diagnostics.py                  |       47 |        4 |       14 |        7 |     82% |47-48, 52, 60, 62-\>65, 67-\>81, 71-\>75, 75-\>81 |
| vpop\_calibration/sdk/model.py                        |       46 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/saem.py                         |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/\_\_init\_\_.py   |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/analytical.py     |       41 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/base.py           |       30 |        1 |        0 |        0 |     97% |        38 |
| vpop\_calibration/structural\_model/gp.py             |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/sbml.py           |       94 |        9 |       14 |        3 |     87% |34-35, 38, 68-73, 76 |
| vpop\_calibration/structural\_model/simwork.py        |      128 |        7 |       22 |        2 |     91% |72-89, 121 |
| vpop\_calibration/utils.py                            |       24 |        3 |        6 |        2 |     83% | 29-30, 35 |
| **TOTAL**                                             | **3406** |  **354** |  **704** |  **129** | **86%** |           |
