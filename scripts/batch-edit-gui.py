from gooey import Gooey

from .batch_edit import main

main = Gooey(program_name="picture-tool")(main)

if __name__ == "__main__":
    main()
