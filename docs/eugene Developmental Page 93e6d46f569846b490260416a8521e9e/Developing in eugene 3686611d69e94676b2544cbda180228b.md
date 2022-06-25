# Developing in eugene

# Tests

- Organize all functionality, make sure we have tests for it ([https://testbook.readthedocs.io/en/latest/index.html](https://testbook.readthedocs.io/en/latest/index.html)) and clean up + organize

## A notebook for each module

## A unit test for each module

## Dummy dataset generation

## Synthetic dataset generation

- Build a simulation framework

    ![EUGENE-46.jpg](Developing%20in%20eugene%203686611d69e94676b2544cbda180228b/EUGENE-46.jpg)

    ![EUGENE-47.jpg](Developing%20in%20eugene%203686611d69e94676b2544cbda180228b/EUGENE-47.jpg)

    1. Start by defining your regulatory grammars (e.g. two ETS in a row)
        1. How many regulatory grammars to generate?
        2. How complex? Layers of complexity? Different simulations with different amounts of complexity?
    2. Next define what contexts the regulatory grammars are active in (e.g. weak neural, neural, ectopic and non-active)
        1. Basically bin each of the grammars defined above into a “regulatory sequence class”
    3. Next we have to define how many sequences fall into these categories (have the real data to help guide the way here)
        1. Want the same total number of sequences
    4. Use the [MPRAnalyze](https://www.notion.so/MPRAnalyze-c5cec6b3d4bd4d5b82c2027f9807e5a1) tool to simulate labels for each of the sequences defined above
        1. Need to choose the parameters for each sequence class
    5. Train different models on the this simulated data
        1. Choose a binarization threshold
        2. Do a regression
