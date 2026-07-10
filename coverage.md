| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| vpop\_calibration/\_\_init\_\_.py                     |        6 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/config.py                           |        6 |        1 |        2 |        1 |     75% |         7 |
| vpop\_calibration/data\_generation.py                 |       91 |        2 |       12 |        4 |     94% |60, 74, 129->132, 168->171 |
| vpop\_calibration/interface.py                        |       27 |        1 |        2 |        1 |     93% |        43 |
| vpop\_calibration/metropolis\_hastings.py             |       36 |        1 |        2 |        1 |     95% |        82 |
| vpop\_calibration/model/\_\_init\_\_.py               |        2 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/model/data.py                       |      162 |       23 |       44 |       14 |     80% |51, 53, 55, 57, 158, 216, 264-274, 348, 355, 362, 367, 371, 375, 382-384, 433 |
| vpop\_calibration/model/gp.py                         |      228 |       49 |       68 |       20 |     75% |62, 67, 103, 140, 142-143, 147-150, 225-226, 233, 253, 287, 309, 311, 365, 386, 390-395, 402-410, 412->347, 436-438, 466, 476-485, 515->exit, 592-599 |
| vpop\_calibration/model/plot.py                       |      124 |       11 |       32 |        8 |     88% |77-78, 160-161, 187->190, 232->191, 237-238, 257, 266->268, 269-272 |
| vpop\_calibration/pynlme/\_\_init\_\_.py              |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/conditional\_distribution.py |      159 |       19 |       36 |        5 |     83% |68, 73->75, 80-83, 167, 188-194, 247-254, 271 |
| vpop\_calibration/pynlme/config.py                    |        7 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/data.py                      |       47 |        0 |        8 |        0 |    100% |           |
| vpop\_calibration/pynlme/diagnostics.py               |      112 |        5 |       14 |        4 |     93% |56-57, 128->131, 174-176, 195->199 |
| vpop\_calibration/pynlme/indexing.py                  |       60 |        0 |        8 |        0 |    100% |           |
| vpop\_calibration/pynlme/model.py                     |      222 |       20 |       26 |        2 |     91% |44->51, 163->159, 529-574 |
| vpop\_calibration/pynlme/params.py                    |      106 |        3 |       16 |        2 |     96% |54, 56, 129 |
| vpop\_calibration/pynlme/plot.py                      |      332 |       33 |      104 |       32 |     85% |36->38, 70, 72-73, 119-120, 129, 152, 185-186, 197->201, 202, 211-212, 233, 268, 283, 287->293, 299, 304, 338, 367, 380, 421, 423, 436-437, 482, 496, 502, 508, 512-513, 521, 639, 648, 650->653, 680->691 |
| vpop\_calibration/pynlme/residuals.py                 |       61 |        2 |       12 |        2 |     95% |   49, 108 |
| vpop\_calibration/pynlme/schemas.py                   |       13 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/utils.py                     |       18 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/config.py                      |       22 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/estimates.py                   |       73 |       17 |       18 |        0 |     70% |92-101, 104-111, 126, 130 |
| vpop\_calibration/saem/m\_step.py                     |       36 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/optimizer.py                   |      129 |       14 |       42 |        7 |     83% |35, 40->52, 104, 111-114, 117-120, 123-128, 213->234, 325-333 |
| vpop\_calibration/saem/plot.py                        |       41 |       32 |       14 |        0 |     16% |17-50, 53-61, 64 |
| vpop\_calibration/saem/scheduler.py                   |       43 |        2 |       18 |        2 |     93% |    55, 70 |
| vpop\_calibration/saem/utils.py                       |       27 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/\_\_init\_\_.py                 |        4 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/diagnostics.py                  |       35 |        0 |        8 |        4 |     91% |35->38, 38->41, 41->44, 44->47 |
| vpop\_calibration/sdk/model.py                        |        9 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/saem.py                         |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/\_\_init\_\_.py   |        5 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/analytical.py     |       41 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/base.py           |       29 |        1 |        0 |        0 |     97% |        37 |
| vpop\_calibration/structural\_model/gp.py             |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/structural\_model/simwork.py        |      127 |        5 |       20 |        2 |     94% |72-83, 118 |
| vpop\_calibration/utils.py                            |        9 |        0 |        2 |        0 |    100% |           |
| **TOTAL**                                             | **2480** |  **241** |  **512** |  **111** | **86%** |           |
