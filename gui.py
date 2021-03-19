from __future__ import generator_stop

from gooey import Gooey

from main import main

main = Gooey(program_name="picture-tool")(main)

if __name__ == "__main__":
	main()
