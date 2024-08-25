from pathlib import Path

import pycc

from tree_ai_box.TreeAIBox import TreeAIBox


class TreeAIBoxCC(pycc.PythonPluginInterface):
    """Define a Plugin for CloudCompare-PythonRuntime."""

    def __init__(self):
        """Construct the object."""
        pycc.PythonPluginInterface.__init__(self)

    def getIcon(self) -> str:
        """Get the path to the plugin icon."""
        return str((Path(__file__).parents[0] / "logo.png").resolve())

    def getActions(self) -> list[pycc.Action]:
        """List of actions exposed by the plugin."""
        return [pycc.Action(name="3D Tree Algorithm Collections", icon=self.getIcon(), target=main)]


def main() -> None:
    """TreeAIBox CloudCompare Plugin main action."""
    mainWindow = TreeAIBox()
    mainWindow.show()
