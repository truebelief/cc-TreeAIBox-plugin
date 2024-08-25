# Standard library imports
import json
import sys
from pathlib import Path

# Third-party imports
import cccorelib
import laspy
import numpy as np
import pycc
import requests
import torch
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QPalette, QPixmap
from PyQt5.QtWidgets import QMessageBox, QStyledItemDelegate

CC = pycc.GetInstance()


class ComboBoxDelegate(QStyledItemDelegate):
    """Custom delegate for combo box items."""

    def __init__(self, parent=None):
        """Initialize the ComboBoxDelegate."""
        super(ComboBoxDelegate, self).__init__(parent)

    def paint(self, painter, option, index):
        """Paint the combo box item."""
        color = index.data(Qt.ForegroundRole)
        if color:
            option.palette.setColor(QPalette.Text, color)
        super(ComboBoxDelegate, self).paint(painter, option, index)


class TreeAIBox(QtWidgets.QMainWindow):
    """Main window for the TreeAIBox application."""

    def __init__(self):
        """Initialize the TreeAIBox application."""
        super(TreeAIBox, self).__init__()
        uic.loadUi(Path(__file__).parent / "form.ui", self)
        self.checkDevice()
        self.btnWoodclsApply.clicked.connect(self.onBtnWoodclsApplyClicked)
        if self.tabWidget is not None:
            self.tabWidget.currentChanged.connect(self.selectedRadioTexts)
        with (Path(__file__).parent / "model_zoo.json").open("r") as f:
            self.MODEL_NAMES = json.load(f)

        self.MODEL_SERVER_PATH = "http://xizhouxin.com/static/treeaibox/"

        self.MODEL_LOCAL_PATH = self.getModelStorageDir()
        self.LOG_LOCAL_PATH = self.getLogStorageDir()
        print("Model location:" + str(self.MODEL_LOCAL_PATH))
        print("Log location:" + str(self.LOG_LOCAL_PATH))

        self.selectedRadioTexts()
        self.registerRadioToggled(self.tabWidget)
        self.registerComboBoxDelegate(self.tabWidget)
        self.tabWidget.currentChanged.connect(self.updateCurrentComboItemColor)
        self.tabWidget.currentChanged.connect(lambda: self.progressBar.setValue(0))

        self.btnWoodclsDownload.clicked.connect(self.onDownloadClicked)
        self.btnWoodclsOpenModelPath.clicked.connect(self.onBtnOpenModelPath)
        self.btnQSMApply.clicked.connect(self.onBtnCreateQSM)
        self.btnQSMInitSegmentation.clicked.connect(self.onBtnQSMInitSegmentation)
        self.btnQSMOpenOutputPath.clicked.connect(self.onBtnOpenLogPath)

        self.updateCurrentComboItemColor()
        self.radioWoodClsBranch.toggled.connect(
            lambda: self.labelWoodClsImage.setPixmap(QPixmap(":/img/woodcls_branch.png"))
        )
        self.radioWoodClsStem.toggled.connect(
            lambda: self.labelWoodClsImage.setPixmap(QPixmap(":/img/woodcls_stem.png"))
        )

    def updateProgress(self, bytes_downloaded, total_size):
        """Update the progress bar."""
        if total_size > 0:
            percent = (bytes_downloaded / total_size) * 100
            self.progressBar.setValue(int(percent))
        else:
            self.progressBar.setValue(0)

    def getModelStorageDir(self):
        """Get the directory for storing models."""
        appdata_dir = Path.home() / "AppData" / "Local"
        model_dir = appdata_dir / "CloudCompare" / "TreeAIBox" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def getLogStorageDir(self):
        """Get the directory for storing logs."""
        appdata_dir = Path.home() / "AppData" / "Local"
        log_dir = appdata_dir / "CloudCompare" / "TreeAIBox" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def downloadFile(self, url, local_path):
        """Download a file from a URL to a local path."""
        start_downloading = True
        if Path(local_path).is_file():
            start_downloading = self.showQuestionMessageBox(
                "Model found in the local folder. Do you want to overwrite?"
            )
        if start_downloading:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # check if downloading is successful
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024
                with Path(local_path).open("wb") as file:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        self.updateProgress(file.tell(), total_size)
                self.showInfoMessagebox(f"File downloaded to {local_path}", "success")
                return True
            except requests.exceptions.RequestException:
                self.showInfoMessagebox("Failed to download file from our server", "error")
                return False
        return True

    def checkDevice(self):
        """Check if CUDA is available and update the UI accordingly."""
        self.iscuda = torch.cuda.is_available()
        if self.iscuda:
            self.checkboxGPU.setChecked(True)
            print("gpu[cuda] available")
        else:
            self.checkboxGPU.setChecked(False)
            self.checkboxGPU.setEnabled(False)
            print("gpu[cuda] unavailable, use cpu instead")

    def showInfoMessagebox(self, text, title):
        """Show an information message box."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def showQuestionMessageBox(self, text):
        """Show a question message box and return the user's response."""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText(text)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)

        return msg_box.exec_() == QMessageBox.Yes

    def registerRadioToggled(self, widget):
        """Register radio button toggle events."""
        for child in widget.findChildren(QtWidgets.QRadioButton):
            child.toggled.connect(self.selectedRadioTexts)
            child.toggled.connect(self.updateCurrentComboItemColor)

    def registerComboBoxDelegate(self, widget):
        """Register combo box delegates."""
        for child in widget.findChildren(QtWidgets.QComboBox):
            child.setCurrentIndex(0)
            delegate = ComboBoxDelegate(child)
            child.setItemDelegate(delegate)

    def updateAnyKeywords(self, group_check, group_widget):
        """Update keywords based on checked radio buttons."""
        if group_check.isChecked():
            keywords_to_remove = [
                child.text() for child in group_widget.findChildren(QtWidgets.QRadioButton) if child.isChecked()
            ]
            self.model_choice_key_words = [
                item for item in self.model_choice_key_words if item not in keywords_to_remove
            ]
            self.updateComboItems()
            self.updateCurrentComboItemColor()
        else:
            self.selectedRadioTexts()

    def selectedRadioTexts(self):
        """Get selected radio button texts."""
        selected_radios = [self.tabWidget.tabText(self.tabWidget.currentIndex())]
        widget = self.tabWidget.currentWidget()
        selected_radios.extend(
            [
                child.text()
                for child in widget.findChildren(QtWidgets.QRadioButton)
                if child.isChecked() and child.isEnabled()
            ]
        )

        self.model_choice_key_words = selected_radios
        self.updateComboItems()

    def updateCurrentComboItemColor(self):
        """Update the color of the current combo box item."""
        combo_box = self.tabWidget.currentWidget().findChild(QtWidgets.QComboBox)
        if combo_box is None:
            return
        current_index = combo_box.currentIndex()
        if current_index >= 0:
            index = combo_box.model().index(current_index, 0)
            color = index.data(Qt.ForegroundRole)
            if color:
                combo_box.setStyleSheet(f"QComboBox {{ color: {color.name()}; }}")

    def updateComboItems(self):
        """Update combo box items based on selected keywords."""
        matching_indices = [
            index
            for index, string in enumerate(self.MODEL_NAMES)
            if all(keyword.lower() in string.lower() for keyword in self.model_choice_key_words)
        ]
        widget = self.tabWidget.currentWidget()
        combo_widget = widget.findChild(QtWidgets.QComboBox)
        if combo_widget is None:
            return
        if matching_indices:
            combo_widget.clear()
            self.available_model_lists = [self.MODEL_NAMES[matching_index] for matching_index in matching_indices]
            for _, model_name in enumerate(self.available_model_lists):
                combo_widget.addItem(model_name)
                index = combo_widget.model().index(combo_widget.count() - 1, 0)
                file_path = self.MODEL_LOCAL_PATH / f"{model_name}.pth"
                color = QColor(Qt.blue) if file_path.exists() else QColor(180, 180, 180)
                combo_widget.model().setData(index, color, Qt.ForegroundRole)
                combo_widget.model().setData(index, str(file_path), Qt.UserRole)
        else:
            combo_widget.clear()

    def loadDemoData(self):
        """Load demo data."""
        path_to_las = Path("data/280_T2_486.laz")
        pcd_basename = path_to_las.stem
        las = laspy.read(path_to_las)
        xs = (las.x - las.header.x_min).astype(pycc.PointCoordinateType)
        ys = (las.y - las.header.y_min).astype(pycc.PointCoordinateType)
        zs = (las.z - las.header.z_min).astype(pycc.PointCoordinateType)
        point_cloud = pycc.ccPointCloud(xs, ys, zs)

        point_cloud.setGlobalShift(-las.header.x_min, -las.header.y_min, -las.header.z_min)
        point_cloud.setName(pcd_basename)

        pcd_wrapper = pycc.ccHObject(pcd_basename)
        pcd_wrapper.addChild(point_cloud)

        CC.addToDB(point_cloud)
        CC.addToDB(pcd_wrapper)
        CC.redrawAll()
        CC.updateUI()

    def onDownloadClicked(self):
        """Handle download button click."""
        model_file = self.tabWidget.currentWidget().findChild(QtWidgets.QComboBox).currentText()
        if self.downloadFile(f"{self.MODEL_SERVER_PATH}/{model_file}.pth", self.MODEL_LOCAL_PATH / f"{model_file}.pth"):
            self.updateComboItems()
            self.updateCurrentComboItemColor()

    def onBtnOpenModelPath(self):
        """Open the model storage directory."""
        if self.MODEL_LOCAL_PATH.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.MODEL_LOCAL_PATH)))
        else:
            print(f"The path {self.MODEL_LOCAL_PATH} is not a valid directory.")

    def onBtnOpenLogPath(self):
        """Open the log storage directory."""
        if self.LOG_LOCAL_PATH.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.LOG_LOCAL_PATH)))
        else:
            print(f"The path {self.LOG_LOCAL_PATH} is not a valid directory")

    def onBtnWoodclsApplyClicked(self):
        """Handle wood classification apply button click."""
        if not CC.haveSelection():
            self.showInfoMessagebox("Please select at least one point cloud to proceed", "Reminder")
            return

        pcs = CC.getSelectedEntities()
        for pc in pcs:
            if type(pc).__name__ != "ccPointCloud":
                self.showInfoMessagebox("Please select a point cloud, not other data types", "Reminder")
                return
            pcd = pc.points()

            current_script_path = Path(__file__).parent
            import sys

            sys.path.append(str(current_script_path))

            self.iscuda = self.checkboxGPU.isChecked()
            print("using gpu[cuda]" if self.iscuda else "using cpu")

            woodcls_mode = "branch" if self.radioWoodClsBranch.isChecked() else "stem"
            config_file = current_script_path / f"modules/woodcls/{woodcls_mode}ClsConfig.json"
            model_path = self.MODEL_LOCAL_PATH / f"{self.comboPretrainWoodCls.currentText()}.pth"

            from modules.woodcls import woodCls

            woodcls_labels = woodCls.applyWoodCls(config_file, pcd, model_path, self.iscuda, self.progressBar)
            self.progressBar.setValue(100)
            woodcls_labels_field = pc.getScalarFieldIndexByName(f"{woodcls_mode}cls")
            if woodcls_labels_field < 0:
                woodcls_labels_field = pc.addScalarField(f"{woodcls_mode}cls")
            sfArray = pc.getScalarField(woodcls_labels_field).asArray()
            sfArray[:] = woodcls_labels
            pc.getScalarField(woodcls_labels_field).computeMinAndMax()
            pc.setCurrentDisplayedScalarField(woodcls_labels_field)

            pc.showSF(True)
            CC.redrawAll()
            CC.updateUI()

    def onBtnQSMInitSegmentation(self):
        """Handle QSM initial segmentation button click."""
        if not CC.haveSelection():
            self.showInfoMessagebox("Please select at least one point cloud to proceed", "Reminder")
            return

        pcs = CC.getSelectedEntities()
        for pc in pcs:
            if type(pc).__name__ != "ccPointCloud":
                self.showInfoMessagebox("Please select a point cloud, not other data types", "Reminder")
                return
            pcd = pc.points()
            n_pts = len(pcd)
            current_script_path = Path(__file__).parent
            import sys

            sys.path.append(str(current_script_path))

            from modules.qsm import applyQSM

            stemcls_field = pc.getScalarFieldIndexByName("stemcls")
            if stemcls_field < 0:
                self.showInfoMessagebox("Please extract stems points by running stemcls first", "Reminder")
                return
            stemcls = pc.getScalarField(stemcls_field).asArray()
            pcd = np.concatenate([pcd, stemcls[:, np.newaxis]], axis=1)
            branchcls_field = pc.getScalarFieldIndexByName("branchcls")
            if branchcls_field >= 0:
                branchcls = pc.getScalarField(branchcls_field).asArray()
                branch_ind = branchcls == 2
                pcd = pcd[branch_ind]

            init_segs_labels = applyQSM.initSegmentation(
                pcd, self.spinQSMStemSpacing.value(), self.spinQSMBranchBandwidth.value(), progress_bar=self.progressBar
            )
            init_segs_field = pc.getScalarFieldIndexByName("init_segs")
            if init_segs_field < 0:
                init_segs_field = pc.addScalarField("init_segs")
            sfArray = pc.getScalarField(init_segs_field).asArray()
            if branchcls_field >= 0:
                sfArray[:] = np.zeros(n_pts)
                sfArray[branch_ind] = init_segs_labels
            else:
                sfArray[:] = init_segs_labels
            pc.getScalarField(init_segs_field).computeMinAndMax()
            pc.setCurrentDisplayedScalarField(init_segs_field)
            pc.showSF(True)

            CC.redrawAll()
            CC.updateUI()

    def onBtnCreateQSM(self):
        """Handle Create QSM button click."""
        if not CC.haveSelection():
            self.showInfoMessagebox("Please select at least one point cloud to proceed", "Reminder")
            return

        pcs = CC.getSelectedEntities()
        for pc in pcs:
            if type(pc).__name__ != "ccPointCloud":
                self.showInfoMessagebox("Please select a point cloud, not other data types", "Reminder")
                return
            pcd = pc.points()
            n_pts = len(pcd)
            current_script_path = Path(__file__).parent
            import sys

            sys.path.append(str(current_script_path))
            from modules.qsm import applyQSM

            stemcls_field = pc.getScalarFieldIndexByName("stemcls")
            if stemcls_field < 0:
                self.showInfoMessagebox("Please extract stems points by running stemcls first", "Reminder")
                return
            stemcls = pc.getScalarField(stemcls_field).asArray()

            init_segs_field = pc.getScalarFieldIndexByName("init_segs")
            if init_segs_field < 0:
                self.showInfoMessagebox("Please run initial segmentation first", "Reminder")
                return
            init_segs = pc.getScalarField(init_segs_field).asArray()

            stemcls = stemcls - np.min(stemcls)
            pcd = np.concatenate([pcd, stemcls[:, np.newaxis], init_segs[:, np.newaxis]], axis=1)

            branchcls_field = pc.getScalarFieldIndexByName("branchcls")
            if branchcls_field >= 0:
                branchcls = pc.getScalarField(branchcls_field).asArray()
                branch_ind = branchcls == 2
                pcd = pcd[branch_ind]
            tree, segs_centroids, segs_labels, tree_centroid_radius = applyQSM.applyQSM(
                pcd,
                max_connectivity_search_distance=self.spinQSMConnectivitySearchRadius.value(),
                occlusion_distance_cutoff=self.spinQSMOcclusionCutoff.value(),
                progress_bar=self.progressBar,
            )
            pcd_wrapper = pycc.ccHObject(f"{pc.getName()}_branch_medial")
            for i, branch in enumerate(tree):
                branch_medial_pcd = pycc.ccPointCloud(f"{pc.getName()}_branch_medial")
                for node in branch:
                    branch_medial_pcd.addPoint(
                        cccorelib.CCVector3(segs_centroids[node][0], segs_centroids[node][1], segs_centroids[node][2])
                    )
                branch_polyline = pycc.ccPolyline(branch_medial_pcd)
                branch_polyline.setClosed(False)
                branch_polyline.addPointIndex(0, branch_medial_pcd.size())
                branch_polyline.setName("Stem" if i == 0 else f"Branch_{i}")
                pcd_wrapper.addChild(branch_polyline)

                CC.addToDB(branch_polyline)
            CC.addToDB(pcd_wrapper)

            obj_path = self.LOG_LOCAL_PATH / f"{pc.getName()}_woodobj.obj"
            xml_path = self.LOG_LOCAL_PATH / f"{pc.getName()}_wood.xml"
            applyQSM.saveTreeToObj(tree_centroid_radius, obj_path)
            applyQSM.saveTreeToXML(tree, tree_centroid_radius, xml_path)
            self.progressBar.setValue(90)
            params = pycc.FileIOFilter.LoadParameters()
            params.parentWidget = CC.getMainWindow()
            CC.loadFile(str(obj_path), params)
            self.progressBar.setValue(100)
            CC.redrawAll()
            CC.updateUI()

            segs_field = pc.getScalarFieldIndexByName("segs")
            if segs_field < 0:
                segs_field = pc.addScalarField("segs")
            sfArray = pc.getScalarField(segs_field).asArray()
            if branchcls_field >= 0:
                sfArray[:] = np.zeros(n_pts)
                sfArray[branch_ind] = segs_labels
            else:
                sfArray[:] = segs_labels
            pc.getScalarField(segs_field).computeMinAndMax()
            pc.setCurrentDisplayedScalarField(segs_field)
            pc.showSF(True)

            CC.redrawAll()
            CC.updateUI()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = TreeAIBox()
    mainWindow.show()
    sys.exit(app.exec_())
