# How to edit GUI

1. `cd` to the current directory.
1. Open `gui.ui` file in `designer`.

    ```sh
    designer gui.ui
    ```

1. Save to `gui.py`.

   ```sh
   pyuic5 -x gui.ui -o gui.py
   ```

1. Substitute old `class Ui_Dialog(object)` in your main GUI file for the newly generated one.
