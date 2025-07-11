import React, { useRef, useState } from 'react';
import { Cropper } from 'react-advanced-cropper';
import 'react-advanced-cropper/dist/style.css';
import './SignatureExtractor.css';

function ImageCropper() {
  const [originalImage, setOriginalImage] = useState(null);
  const [croppedImage, setCroppedImage] = useState(null);
  const [originalSize, setOriginalSize] = useState(0);
  const [croppedSize, setCroppedSize] = useState(0);
  const [showCropper, setShowCropper] = useState(false);

  const cropperRef = useRef(null);

  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      setOriginalImage(reader.result);
      setOriginalSize((file.size / 1024).toFixed(2));
      setCroppedImage(null);
      setShowCropper(true);
    };
    reader.readAsDataURL(file);
  };

  const processImage = (img, callback) => {
  const cv = window.cv;
  if (!cv || !cv.Mat) {
    alert("OpenCV error");
    return;
  }

  // read
  let src = cv.imread(img); 
  let rgbaPlanes = new cv.MatVector();
  cv.split(src, rgbaPlanes);

  const correctedPlanes = new cv.MatVector();
  for (let i = 0; i < 3; i++) { // RGB only
    const plane = rgbaPlanes.get(i);
    const kernel = cv.Mat.ones(7, 7, cv.CV_8U);
    const dilated = new cv.Mat();
    cv.dilate(plane, dilated, kernel);

    const bg = new cv.Mat();
    cv.medianBlur(dilated, bg, 23);

    const diff = new cv.Mat();
    cv.absdiff(plane, bg, diff);
    cv.subtract(new cv.Mat(plane.rows, plane.cols, cv.CV_8U, new cv.Scalar(255)), diff, diff);

    const norm = new cv.Mat();
    cv.normalize(diff, norm, 0, 255, cv.NORM_MINMAX);

    correctedPlanes.push_back(norm);

    plane.delete(); dilated.delete(); bg.delete(); diff.delete(); kernel.delete();
  }

  // brightness and alpha channel
  correctedPlanes.push_back(rgbaPlanes.get(3));
  let corrected = new cv.Mat();
  cv.merge(correctedPlanes, corrected);

  rgbaPlanes.delete();
  correctedPlanes.delete();
  src.delete();

  // grayscale
  const gray = new cv.Mat();
  cv.cvtColor(corrected, gray, cv.COLOR_RGBA2GRAY);

  const thresh = new cv.Mat();
  cv.threshold(gray, thresh, 200, 255, cv.THRESH_BINARY_INV);

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  // cropping
  let xMin = thresh.cols, yMin = thresh.rows, xMax = 0, yMax = 0;
  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i);
    const area = cv.contourArea(cnt);
    if (area > 200) {
      const rect = cv.boundingRect(cnt);
      xMin = Math.min(xMin, rect.x);
      yMin = Math.min(yMin, rect.y);
      xMax = Math.max(xMax, rect.x + rect.width);
      yMax = Math.max(yMax, rect.y + rect.height);
    }
    cnt.delete();
  }

  const pad = 20;
  xMin = Math.max(xMin - pad, 0);
  yMin = Math.max(yMin - pad, 0);
  xMax = Math.min(xMax + pad, thresh.cols);
  yMax = Math.min(yMax + pad, thresh.rows);

  let cropped;
  if (xMax > xMin && yMax > yMin) {
    const rect = new cv.Rect(xMin, yMin, xMax - xMin, yMax - yMin);
    cropped = thresh.roi(rect);
  } else {
    cropped = thresh.clone();
  }

  const binMask = new cv.Mat();
  cv.threshold(cropped, binMask, 100, 255, cv.THRESH_BINARY);

  const rgba = new cv.Mat();
  cv.cvtColor(binMask, rgba, cv.COLOR_GRAY2RGBA);

  for (let i = 0; i < rgba.rows; i++) {
    for (let j = 0; j < rgba.cols; j++) {
      const pixel = rgba.ucharPtr(i, j);
      if (pixel[0] > 0) {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
        pixel[3] = 255;
      } else {
        pixel[3] = 0;
      }
    }
  }

  const resized = new cv.Mat();
  const dsize = new cv.Size(300, 100);
  cv.resize(rgba, resized, dsize, 0, 0, cv.INTER_AREA);

  const canvas = document.createElement("canvas");
  cv.imshow(canvas, resized);

  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    callback(url, blob.size);
  }, 'image/png');

  corrected.delete(); gray.delete(); thresh.delete(); contours.delete(); hierarchy.delete();
  cropped.delete(); binMask.delete(); rgba.delete(); resized.delete();
};



const cropImage = () => {
  const canvas = cropperRef.current?.getCanvas();
  if (!canvas) return;

  const imgElement = new Image();
  imgElement.src = canvas.toDataURL();

  imgElement.onload = () => {
    processImage(imgElement, (url, size) => {
      setCroppedImage(url);
      setCroppedSize((size / 1024).toFixed(2)); // KB
      setShowCropper(false);
    });
  };
};
  

  return (
    <div className="container">
      <input
        type="file"
        accept="image/*"
        onChange={onFileChange}
        style={{ display: 'none' }}
        id="fileInput"
      />
      <button onClick={() => document.getElementById('fileInput').click()} className="upload-button">
        Upload Picture
      </button>

      {showCropper && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Crop Image</h2>
            <Cropper
              src={originalImage}
              ref={cropperRef}
              className="cropper"
              stencilProps={{
                movable: true,
                resizable: true,
              }}
              style={{ height: 400, width: '100%' }}
            />
            <div className="modal-buttons">
              <button className = "modal-button" onClick={() => setShowCropper(false)}>Cancel</button>
              <button className = "modal-button" onClick={cropImage}>Choose</button>
            </div>
          </div>
        </div>
      )}

      {originalImage && croppedImage && (
        <>  
        <div className="image-comparison">
          <div className="image-block">
            <h3>Original Image ({originalSize} KB)</h3>
            <img src={originalImage} alt="Original" />
          </div>
          <div className="image-block">
            <h3>Cropped Image ({croppedSize} KB)</h3>
            <img src={croppedImage} alt="Cropped" />
          </div>
        </div>
        <div style={{ marginTop: '1rem' }}>
          <button class = "recropper" onClick={() => setShowCropper(true)}>Re-crop</button>
            <button class = "upload" onClick={() => `` }>Upload</button>
        </div>
        </>
      )}
    </div>
  );
}

export default ImageCropper;
