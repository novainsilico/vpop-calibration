| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| vpop\_calibration/api/\_\_init\_\_.py                 |        8 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/api/interface.py                    |       53 |       14 |        0 |        0 |     74% |67-84, 87-89, 98-100 |
| vpop\_calibration/compatibility.py                    |       14 |       12 |        0 |        0 |     14% |      3-19 |
| vpop\_calibration/config.py                           |        9 |        1 |        2 |        1 |     82% |         7 |
| vpop\_calibration/data\_generation.py                 |       91 |       64 |       12 |        0 |     26% |30-31, 38-63, 72-82, 91-97, 110-118, 128-157, 167-216 |
| vpop\_calibration/metropolis\_hastings.py             |       54 |       10 |        8 |        1 |     76% |34-51, 54-66, 133 |
| vpop\_calibration/model/\_\_init\_\_.py               |        2 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/model/data.py                       |      162 |      138 |       44 |        0 |     12% |46-209, 233-255, 261-266, 271-276, 283-293, 298-304, 311-337, 361-410, 430-453 |
| vpop\_calibration/model/gp.py                         |      227 |      195 |       68 |        0 |     11% |55-142, 145-153, 216-291, 297-417, 431-438, 444-453, 457-468, 475-484, 488-492, 497-524, 546-551, 561-562, 569-574, 583-584, 589-596 |
| vpop\_calibration/model/plot.py                       |      122 |      110 |       32 |        0 |      8% |24-78, 85-149, 159-226, 231-244, 252-259 |
| vpop\_calibration/pynlme/\_\_init\_\_.py              |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/pynlme/conditional\_distribution.py |      185 |       91 |       44 |        6 |     45% |9-10, 13-14, 34, 42-51, 103, 110-125, 131, 134, 136-\>138, 141, 143-148, 210-236, 240-263, 267-271, 275-276, 280-290, 294-304, 308-316, 323-328, 332-335 |
| vpop\_calibration/pynlme/config.py                    |       14 |        1 |        0 |        0 |     93% |        19 |
| vpop\_calibration/pynlme/data.py                      |       61 |       13 |       12 |        2 |     77% |44-74, 162 |
| vpop\_calibration/pynlme/diagnostics.py               |      164 |      122 |       22 |        0 |     23% |59-66, 83-146, 157-223, 226-285, 288-299, 303-314, 323-389, 394-401 |
| vpop\_calibration/pynlme/error\_estimation.py         |       38 |       15 |       10 |        2 |     65% |17-31, 58, 64-68 |
| vpop\_calibration/pynlme/importance\_sampling.py      |       56 |       31 |       10 |        2 |     41% |21-22, 24, 34-48, 53-63, 69-73, 76-81, 85-101 |
| vpop\_calibration/pynlme/indexing.py                  |       66 |       14 |        8 |        0 |     76% |   118-147 |
| vpop\_calibration/pynlme/initial\_estimates.py        |       78 |       60 |       22 |        0 |     18% |15-37, 45-70, 73-76, 82-100, 113-123, 126-131, 134-140 |
| vpop\_calibration/pynlme/model.py                     |      267 |       39 |       28 |        4 |     84% |42-50, 53-57, 83-\>90, 217-229, 242-\>238, 256-261, 463, 646-691, 694-700 |
| vpop\_calibration/pynlme/params.py                    |      180 |       19 |       48 |       12 |     84% |54, 56, 110-115, 120, 125, 126-\>129, 128, 185, 205, 214, 226, 238-\>234, 253-255, 276, 283-284 |
| vpop\_calibration/pynlme/plot.py                      |      435 |      408 |      134 |        0 |      5% |33-126, 133-194, 205-282, 294-385, 395-501, 510-540, 557-658, 664-747, 757-846, 854-930 |
| vpop\_calibration/pynlme/residuals.py                 |       91 |       18 |        4 |        2 |     79% |42, 101-102, 166-205, 299-309 |
| vpop\_calibration/pynlme/schemas.py                   |       25 |        1 |        2 |        1 |     93% |        19 |
| vpop\_calibration/pynlme/utils.py                     |       26 |        4 |        2 |        0 |     79% |     82-85 |
| vpop\_calibration/saem/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/saem/config.py                      |       29 |        1 |        0 |        0 |     97% |        51 |
| vpop\_calibration/saem/estimates.py                   |       97 |       25 |       26 |        2 |     68% |30, 40-55, 78, 138-139, 158-167, 170-177, 197, 201 |
| vpop\_calibration/saem/fixed\_effects.py              |       26 |        0 |        2 |        0 |    100% |           |
| vpop\_calibration/saem/m\_step.py                     |       53 |        8 |        2 |        0 |     82% |69-84, 87-96 |
| vpop\_calibration/saem/optimizer.py                   |      140 |       29 |       42 |       10 |     73% |37, 42-\>54, 121, 129-148, 155, 162-165, 168-171, 174-181, 273-\>303, 380-381, 392, 397-405 |
| vpop\_calibration/saem/plot.py                        |       47 |       36 |       14 |        0 |     18% |3-4, 7-8, 24-57, 60-68, 71 |
| vpop\_calibration/saem/scheduler.py                   |       48 |        3 |       14 |        0 |     95% |     85-94 |
| vpop\_calibration/saem/utils.py                       |       27 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/sdk/\_\_init\_\_.py                 |        5 |        5 |        0 |        0 |      0% |      1-14 |
| vpop\_calibration/sdk/config.py                       |        7 |        7 |        0 |        0 |      0% |      1-10 |
| vpop\_calibration/sdk/diagnostics.py                  |       43 |       43 |       12 |        0 |      0% |      1-80 |
| vpop\_calibration/sdk/model.py                        |       46 |       46 |        0 |        0 |      0% |     1-147 |
| vpop\_calibration/sdk/saem.py                         |        5 |        5 |        0 |        0 |      0% |      1-10 |
| vpop\_calibration/structural\_model/\_\_init\_\_.py   |        0 |        0 |        0 |        0 |    100% |           |
| vpop\_calibration/structural\_model/analytical.py     |       41 |        1 |        2 |        1 |     95% |        33 |
| vpop\_calibration/structural\_model/base.py           |       30 |       16 |        0 |        0 |     47% | 38, 43-75 |
| vpop\_calibration/structural\_model/gp.py             |       26 |       19 |        2 |        0 |     25% |17-30, 39-58 |
| vpop\_calibration/structural\_model/sbml.py           |       90 |       73 |       12 |        0 |     17% |21-35, 47-118, 130-174, 177-192, 199-230 |
| vpop\_calibration/structural\_model/simwork.py        |      128 |       94 |       22 |        0 |     23% |40-55, 67-91, 99-130, 135-156, 161-190, 200-256, 268-312, 322-348, 355-391 |
| vpop\_calibration/utils.py                            |       24 |        6 |        6 |        2 |     67% |14, 29-30, 34-36 |
| **TOTAL**                                             | **3340** | **1797** |  **668** |   **48** | **42%** |           |
