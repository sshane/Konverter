package:
	poetry build

clean:
	git clean -fxd --exclude .python-version --exclude .vscode
