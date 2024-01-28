# How to contribute

I'm really glad you're reading this, because we need volunteer developers to help this project come to fruition.

Here are some important things to check when you contribute:

  * Please make sure that you write tests.
  * Update the documentation.
  * Add the new dataset to [README.md](./README.md).
  * If your contribution is a paper, please cite this project properly.

## Testing

PoSeidon-dataset's testing is located under `tests/`. 
To run the tests, simply run `pytest` in the root directory of the project.

## Submitting changes

Please send a [GitHub Pull Request to PoSeidon-dataset](https://github.com/PoSeiDon-Workflows/flowbench/pulls) with a 
clear list of what you've done (read more about [pull requests](http://help.github.com/pull-requests/)). 
When you send a pull request, we will love you forever if you include tests.

Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes
should look like this:

    $ git commit -m "A brief summary of the commit
    > 
    > A paragraph describing what changed and its impact."

## Coding conventions

Start reading our code and you'll get the hang of it. We optimize for readability:

  * We indent using four spaces (soft tabs)
  * We use [flake8](https://flake8.pycqa.org/en/latest/) to lint our code.
    * ignore `E501` (max line length)
  * We use [autopep8](https://pypi.org/project/autopep8/) to format our code.
    * ignore `E402` (module level import not at top of file)
    * `--max-line-length=120`

Thanks,
PoSeidon Team