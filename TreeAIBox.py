from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import  QStyledItemDelegate
from PyQt5.QtGui import QColor,QPalette
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtGui import QPixmap

import os
from pathlib import Path
import sys

import pycc
import cccorelib

import laspy
import torch
import numpy as np

import json
import requests

# https://github.com/CloudCompare/CloudCompare/blob/master/CONTRIBUTING.md
# requirements:
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install scikit-image==0.22.0, timm,numpy_groupies,numpy_indexed,json,sklearn,circle-fit

CC = pycc.GetInstance()
from PyQt5.QtWidgets import QMessageBox

class ComboBoxDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super(ComboBoxDelegate, self).__init__(parent)

    def paint(self, painter, option, index):
        color = index.data(Qt.ForegroundRole)
        if color:
            option.palette.setColor(QPalette.Text, color)
        super(ComboBoxDelegate, self).paint(painter, option, index)


class TreeAIBox(QtWidgets.QMainWindow):
    def __init__(self):
        super(TreeAIBox, self).__init__()
        uic.loadUi(os.path.join(Path(__file__).parent.resolve(),'form.ui'),self)
        self.checkDevice()
        self.btnWoodclsApply.clicked.connect(self.onBtnWoodclsApplyClicked)
        if self.tabWidget is not None:
            self.tabWidget.currentChanged.connect(self.selectedRadioTexts)
        with open(os.path.join(Path(__file__).parent,"model_zoo.json"), 'r') as f:
            self.MODEL_NAMES = json.load(f)

        self.MODEL_SERVER_PATH="http://xizhouxin.com/static/treeaibox/"

        self.MODEL_LOCAL_PATH = self.getModelStorageDir()
        self.LOG_LOCAL_PATH = self.getLogStorageDir()
        print("Model location:" + self.MODEL_LOCAL_PATH)
        print("Log location:" + self.LOG_LOCAL_PATH)

        self.selectedRadioTexts()
        self.registerRadioToggled(self.tabWidget)
        self.registerComboBoxDelegate(self.tabWidget)
        self.tabWidget.currentChanged.connect(self.updateCurrentComboItemColor)
        self.tabWidget.currentChanged.connect(lambda : self.progressBar.setValue(0))

        self.btnWoodclsDownload.clicked.connect(self.onDownloadClicked)
        self.btnWoodclsOpenModelPath.clicked.connect(self.onBtnOpenModelPath)
        self.btnQSMApply.clicked.connect(self.onBtnCreateQSM)
        self.btnQSMInitSegmentation.clicked.connect(self.onBtnQSMInitSegmentation)
        self.btnQSMOpenOutputPath.clicked.connect(self.onBtnOpenLogPath)

        self.updateCurrentComboItemColor()
        self.radioWoodClsBranch.toggled.connect(lambda: self.labelWoodClsImage.setPixmap(QPixmap(os.path.join(Path(__file__).parent.resolve(), "img/woodcls_branch.png"))))
        self.radioWoodClsStem.toggled.connect(lambda: self.labelWoodClsImage.setPixmap(QPixmap(os.path.join(Path(__file__).parent.resolve(), "img/woodcls_stem.png"))))


    def updateProgress(self, bytes_downloaded, total_size):
        if total_size > 0:
            percent = (bytes_downloaded / total_size) * 100
            self.progressBar.setValue(int(percent))
        else:
            self.progressBar.setValue(0)

    def getModelStorageDir(self):
        appdata_dir = Path(os.getenv('LOCALAPPDATA'))
        model_dir = appdata_dir / 'CloudCompare' / 'TreeAIBox' / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir)

    def getLogStorageDir(self):
        appdata_dir = Path(os.getenv('LOCALAPPDATA'))
        log_dir = appdata_dir / 'CloudCompare' / 'TreeAIBox' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir)

    def downloadFile(self,url, local_path):
        start_downloading=True
        if os.path.isfile(local_path):
            start_downloading=self.showQuestionMessageBox("Model found in the local folder. Do you want to overwrite?")
        if start_downloading:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # check if downloading is successful
                with open(local_path, 'wb') as out_file:
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024
                    with open(local_path, 'wb') as file:
                        for data in response.iter_content(block_size):
                            file.write(data)
                            self.updateProgress(file.tell(), total_size)
                self.showInfoMessagebox(f"File downloaded to {local_path}","success")
                return True
            except requests.exceptions.RequestException as e:
                self.showInfoMessagebox(f"Failed to download file from our server","error")
                return False
        return True

    def checkDevice(self):
        # Check if CUDA is available
        self.iscuda = torch.cuda.is_available()
        if self.iscuda:
            self.checkboxGPU.setChecked(True)
            print("gpu[cuda] available")
        else:
            self.checkboxGPU.setChecked(False)
            self.checkboxGPU.setEnabled(False)
            print("gpu[cuda] unavailable, use cpu instead")

    def showInfoMessagebox(self,text,title):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()

    def showQuestionMessageBox(self,text):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText(text)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)

        response = msg_box.exec()

        if response == QMessageBox.Yes:
            return True
        else:
            return False


    def registerRadioToggled(self, widget):
        for child in widget.findChildren(QtWidgets.QRadioButton):
            child.toggled.connect(self.selectedRadioTexts)
            child.toggled.connect(self.updateCurrentComboItemColor)
    def registerComboBoxDelegate(self,widget):
        for child in widget.findChildren(QtWidgets.QComboBox):
            child.setCurrentIndex(0)
            delegate = ComboBoxDelegate(child)
            child.setItemDelegate(delegate)

    def updateAnyKeywords(self,group_check,group_widget):
        if group_check.isChecked():
            keywords_to_remove=[]
            for child in group_widget.findChildren(QtWidgets.QRadioButton):
                if child.isChecked():
                    keywords_to_remove.append(child.text())
            self.model_choice_key_words = [item for item in self.model_choice_key_words if item not in keywords_to_remove]
            self.updateComboItems()
            self.updateCurrentComboItemColor()
        else:
            self.selectedRadioTexts()
    def selectedRadioTexts(self):
        selected_radios = [self.tabWidget.tabText(self.tabWidget.currentIndex())]
        widget=self.tabWidget.currentWidget()
        for child in widget.findChildren(QtWidgets.QRadioButton):
            if child.isChecked() and child.isEnabled():
                selected_radios.append(child.text())

        self.model_choice_key_words =selected_radios
        self.updateComboItems()
        return

    def updateCurrentComboItemColor(self):
        combo_box=self.tabWidget.currentWidget().findChild(QtWidgets.QComboBox)
        if combo_box is None:
            return
        current_index = combo_box.currentIndex()
        if current_index >= 0:
            index = combo_box.model().index(current_index, 0)
            color = index.data(Qt.ForegroundRole)
            if color:
                combo_box.setStyleSheet(f"QComboBox {{ color: {color.name()}; }}")
    def updateComboItems(self):
        matching_indices = [index for index, string in enumerate(self.MODEL_NAMES) if all(keyword.lower() in string.lower() for keyword in self.model_choice_key_words)]
        widget = self.tabWidget.currentWidget()
        combo_widget = widget.findChild(QtWidgets.QComboBox)
        if combo_widget is None:
            return
        if len(matching_indices)>0:
            combo_widget.clear()
            self.available_model_lists=[self.MODEL_NAMES[matching_index] for matching_index in matching_indices]
            for k in range(len(self.available_model_lists)):
                combo_widget.addItem(self.available_model_lists[k])
                index = combo_widget.model().index(combo_widget.count() - 1, 0)
                file_path=os.path.join(self.MODEL_LOCAL_PATH, self.available_model_lists[k] + ".pth")
                color = QColor(Qt.blue) if os.path.exists(file_path) else QColor(180, 180, 180)
                combo_widget.model().setData(index, color, Qt.ForegroundRole)
                combo_widget.model().setData(index, file_path, Qt.UserRole)
        else:
            combo_widget.clear()


    def loadDemoData(self):
        path_to_las = r"data\280_T2_486.laz"
        pcd_basename=os.path.basename(path_to_las)[:-4]
        las = laspy.read(path_to_las)
        xs = (las.x - las.header.x_min).astype(pycc.PointCoordinateType)
        ys = (las.y - las.header.y_min).astype(pycc.PointCoordinateType)
        zs = (las.z - las.header.z_min).astype(pycc.PointCoordinateType)
        point_cloud = pycc.ccPointCloud(xs, ys, zs)

        # Add the global shift to CloudCompare so that it can use it
        point_cloud.setGlobalShift(-las.header.x_min, -las.header.y_min, -las.header.z_min)
        point_cloud.setName(pcd_basename)

        # point_cloud.addScalarField('stemcls')
        # stem_scalar_index=point_cloud.getScalarFieldIndexByName("stemcls")
        # stem_scalar = point_cloud.getScalarField(stem_scalar_index)
        # sfArray=stem_scalar.asArray()
        # sfArray[:] = las.stemcls
        # stem_scalar.computeMinAndMax()  # must call this before the scalar field is updated in CC
        # point_cloud.setCurrentDisplayedScalarField(stem_scalar_index)
        # point_cloud.showSF(True)
        #
        pcd_wrapper = pycc.ccHObject(pcd_basename)
        pcd_wrapper.addChild(point_cloud)

        CC.addToDB(point_cloud)
        CC.addToDB(pcd_wrapper)
        CC.redrawAll()
        CC.updateUI()

    def onDownloadClicked(self):
        model_file = self.tabWidget.currentWidget().findChild(QtWidgets.QComboBox).currentText()
        # self.showInfoMessagebox(model_file,"reminder")
        if self.downloadFile(os.path.join(self.MODEL_SERVER_PATH, model_file + ".pth"),
                              os.path.join(self.MODEL_LOCAL_PATH, model_file + ".pth")):
            self.updateComboItems()
            self.updateCurrentComboItemColor()

    def onBtnOpenModelPath(self):
        if os.path.isdir(self.MODEL_LOCAL_PATH):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.MODEL_LOCAL_PATH))
        else:
            print(f"The path {self.MODEL_LOCAL_PATH} is not a valid directory.")

    def onBtnOpenLogPath(self):
        if os.path.isdir(self.LOG_LOCAL_PATH):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.LOG_LOCAL_PATH))
        else:
            print(f"The path {self.LOG_LOCAL_PATH} is not a valid directory")

    def onBtnWoodclsApplyClicked(self):
        if not CC.haveSelection():
            self.show_info_messagebox('Please select at least one point cloud to proceed', "Reminder")
        else:
            pcs = CC.getSelectedEntities()
            for pc in pcs:
                if type(pc).__name__!='ccPointCloud':#ccPointCloud(point cloud),ccHObject(grouped)
                    self.show_info_messagebox("Please select a point cloud, not other data types", "Reminder")
                    return
                pcd = pc.points()

                current_script_path=os.path.dirname(os.path.realpath(__file__))
                sys.path.append(current_script_path)

                self.iscuda = self.checkboxGPU.isChecked()
                if self.iscuda:
                    print("using cpu")
                else:
                    print("using gpu[cuda]")

                if self.radioWoodClsBranch.isChecked():
                    woodcls_mode = "branch"
                else:
                    woodcls_mode = "stem"

                config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'modules\\woodcls\\{self.comboPretrainWoodCls.currentText()}.json')
                model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.MODEL_LOCAL_PATH + "\\" + str(self.comboPretrainWoodCls.currentText()) + ".pth")

                from modules.woodcls import woodCls
                woodcls_labels = woodCls.applyWoodCls(config_file, pcd, model_path, self.iscuda, self.progressBar)
                self.progressBar.setValue(100)
                woodcls_labels_field = pc.getScalarFieldIndexByName(f"{woodcls_mode}cls")
                if woodcls_labels_field < 0:
                    woodcls_labels_field = pc.addScalarField(f"{woodcls_mode}cls")
                sfArray = pc.getScalarField(woodcls_labels_field).asArray()
                sfArray[:] = woodcls_labels
                pc.getScalarField(woodcls_labels_field).computeMinAndMax()  # must call this before the scalar field is updated in CC
                pc.setCurrentDisplayedScalarField(woodcls_labels_field)

                pc.showSF(True)
                CC.redrawAll()
                CC.updateUI()
        return

    def onBtnQSMInitSegmentation(self):
        if not CC.haveSelection():
            self.showInfoMessagebox('Please select at least one point cloud to proceed', "Reminder")
            return
        else:
            pcs = CC.getSelectedEntities()
        for pc in pcs:
            if type(pc).__name__ != 'ccPointCloud':  # ccPointCloud(point cloud),ccHObject(grouped)
                self.showInfoMessagebox("Please select a point cloud, not other data types", "Reminder")
                return
            pcd = pc.points()
            n_pts = len(pcd)
            current_script_path = os.path.dirname(os.path.realpath(__file__))
            sys.path.append(current_script_path)

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

            init_segs_labels = applyQSM.initSegmentation(pcd,self.spinQSMStemSpacing.value(),self.spinQSMBranchBandwidth.value(),progress_bar=self.progressBar)
            init_segs_field = pc.getScalarFieldIndexByName("init_segs")
            if init_segs_field < 0:
                init_segs_field = pc.addScalarField("init_segs")
            sfArray = pc.getScalarField(init_segs_field).asArray()
            if branchcls_field >= 0:
                sfArray[:] = np.zeros(n_pts)
                sfArray[branch_ind] = init_segs_labels
            else:
                sfArray[:] = init_segs_labels
            pc.getScalarField(init_segs_field).computeMinAndMax()  # must call this before the scalar field is updated in CC
            pc.setCurrentDisplayedScalarField(init_segs_field)
            pc.showSF(True)

            CC.redrawAll()
            CC.updateUI()

    def onBtnCreateQSM(self):
        if not CC.haveSelection():
            self.showInfoMessagebox('Please select at least one point cloud to proceed', "Reminder")
            return
        else:
            pcs = CC.getSelectedEntities()
        for pc in pcs:
            if type(pc).__name__ != 'ccPointCloud':  # ccPointCloud(point cloud),ccHObject(grouped)
                self.showInfoMessagebox("Please select a point cloud, not other data types", "Reminder")
                return
            pcd = pc.points()
            n_pts=len(pcd)
            current_script_path = os.path.dirname(os.path.realpath(__file__))
            sys.path.append(current_script_path)
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

            stemcls=stemcls-np.min(stemcls)
            pcd=np.concatenate([pcd, stemcls[:, np.newaxis],init_segs[:,np.newaxis]], axis=1)

            branchcls_field = pc.getScalarFieldIndexByName("branchcls")
            if branchcls_field>=0:
                branchcls = pc.getScalarField(branchcls_field).asArray()
                branch_ind=branchcls == 2
                pcd=pcd[branch_ind]
            tree, segs_centroids, segs_labels,tree_centroid_radius = applyQSM.applyQSM(pcd,max_connectivity_search_distance=self.spinQSMConnectivitySearchRadius.value(), occlusion_distance_cutoff=self.spinQSMOcclusionCutoff.value(),
                                                                                       progress_bar=self.progressBar)
            pcd_wrapper = pycc.ccHObject(f"{pc.getName()}_branch_medial")
            for i,branch in enumerate(tree):
                branch_medial_pcd = pycc.ccPointCloud(f"{pc.getName()}_branch_medial")
                for node in branch:
                    branch_medial_pcd.addPoint(cccorelib.CCVector3(segs_centroids[node][0], segs_centroids[node][1], segs_centroids[node][2]))
                branch_polyline=pycc.ccPolyline(branch_medial_pcd)
                branch_polyline.setClosed(False)
                branch_polyline.addPointIndex(0, branch_medial_pcd.size())
                if i==0:
                    branch_polyline.setName(f"Stem")
                else:
                    branch_polyline.setName(f"Branch_{i}")
                pcd_wrapper.addChild(branch_polyline)

                CC.addToDB(branch_polyline)
            CC.addToDB(pcd_wrapper)

            obj_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),self.LOG_LOCAL_PATH + "\\" + f"{pc.getName()}_woodobj.obj")
            xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),self.LOG_LOCAL_PATH + "\\" + f"{pc.getName()}_wood.xml")
            applyQSM.saveTreeToObj(tree_centroid_radius,obj_path)
            xml_output = applyQSM.saveTreeToXML(tree, tree_centroid_radius, xml_path)
            self.progressBar.setValue(90)
            params = pycc.FileIOFilter.LoadParameters()
            params.parentWidget = CC.getMainWindow()
            obj = CC.loadFile(obj_path, params)
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
            pc.getScalarField(segs_field).computeMinAndMax()  # must call this before the scalar field is updated in CC
            pc.setCurrentDisplayedScalarField(segs_field)
            pc.showSF(True)

            CC.redrawAll()
            CC.updateUI()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = TreeAIBox()
    mainWindow.show()
    # sys.exit(app.exec_())


