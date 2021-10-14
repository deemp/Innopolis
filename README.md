# BS19-F21-TM

Danko Danila's repo

## Homeworks

1. [HW1](https://colab.research.google.com/drive/1PP9_DvqwEvWMQBGfOrU7PobwpEPjV-HC?usp=sharing)
2. [HW2](./HW2)
3. [Big HW 1](./Big-HW-1)
4. [Big HW 2](./Big-HW-2)

## Installation

1. Open a terminal
   - Linux: `Ctrl+Alt+T`
   - Windows: `Win+X` -> Windows PowerShell (Admin) -> Press `Enter`
   - MacOS (not all screen models supported): `Cmd+Space` -> Type `terminal` -> Press `Enter`

1. Install `make`
   * Linux

      ```sh
      sudo apt-get install build-essential
      ```

   * Windows

      ```sh
      choco install make
      ```

   * MacOS

      ```sh
      brew install make
      ```

1. Navigate to Desktop via `cd`. You may also `cd` to another directory, of course.

   ```sh
   cd ~/Desktop
   ```

1. Clone the project there

   ```sh
   git clone https://github.com/br4ch1st0chr0n3/TM
   ```

1. Navigate to the project root folder

   ```sh
   cd TM
   ```

## Run tasks
   * On Windows, use `PowerShell`

   ```sh
   make TASK=task_name run 
   ```

For specific tasks change `task_name` to one of the following constants
   * HW2, Task 1 -> `HW2_1`
   * Big HW 1 -> `BIG_1`

## Virtual environment
   * **Activate**
      * Windows
         ```sh
         env\Scripts\activate.ps1
         ```
      * Linux / MacOS
         ```sh
         source env/Scripts/activate
         ```
   
   * **Deactivate**
   ```sh
   deactivate
   ```

   * **Remove**
   ```sh
   rm -r env
   ```

## Notes
1. For [Control bootcamp](https://www.youtube.com/playlist?list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m) - <img src="https://cdn.mathcha.io/resources/logo.png" width="20" title="hover text">[link](https://www.mathcha.io/editor/Ov4BQso6UzgsgZHgxEJL2T0EWMXfvzJ8d3trKYj04)
