import React, { useState, useRef } from "react";

export default function SignatureExtractor() {
  const [processedImage, setProcessedImage] = useState(null);
  const [origSize, setOrigSize] = useState(null);
  const [procSize, setProcSize] = useState(null);
  const inputRef = useRef(null);

  const processImage = (img, file) => {
    const cv = window.cv;
    if (!cv || !cv.Mat) {
      alert("OpenCV error");
      return;
    }

    const src = cv.imread(img);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const thresh = new cv.Mat();
    cv.threshold(gray, thresh, 120, 255, cv.THRESH_BINARY_INV);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let xMin = thresh.cols,
      yMin = thresh.rows,
      xMax = 0,
      yMax = 0;
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
    cv.threshold(cropped, binMask, 127, 255, cv.THRESH_BINARY);

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
    //data location
    const dataUrl = canvas.toDataURL("image/png");
    setProcessedImage(dataUrl);

    if (file) {
      setOrigSize((file.size / 1024).toFixed(2));
    } else {
      setOrigSize(null);
    }
    // Show processed image size in KB
    const base64Length = dataUrl.length - (dataUrl.indexOf(",") + 1);
    const processedBytes = Math.floor((base64Length * 3) / 4);
    setProcSize((processedBytes / 1024).toFixed(2));

    src.delete();
    gray.delete();
    thresh.delete();
    contours.delete();
    hierarchy.delete();
    binMask.delete();
    rgba.delete();
    resized.delete();
    if (cropped) cropped.delete();
  };

  return (
    <div className="flex flex-col items-center p-4">
      <h1 className="text-2xl font-bold mb-4">Signature Extractor with OpenCV.js</h1>
      <input
        type="file"
        accept="image/*"
        ref={inputRef}
        className="mb-4"
        onChange={(e) => {
          const file = e.target.files[0];
          if (!file) return;

          const img = new window.Image();
          img.src = URL.createObjectURL(file);
          img.onload = () => processImage(img, file);
        }}
      />
      {processedImage && (
        <div className="border rounded shadow p-4">
          <h2 className="text-lg font-semibold mb-2">Processed Image:</h2>
          <img
            src={processedImage}
            alt="Extracted Signature"
            className="w-[300px] h-[100px]"
          />
        </div>
      )}
      {origSize && procSize && (
        <div className="mt-4 text-center">
          <p>
            Original Size:{" "}
            <span className="font-medium">{origSize} KB</span>
          </p>
          <p>
            Processed Size:{" "}
            <span className="font-medium">{procSize} KB</span>
          </p>
        </div>
      )}
    </div>
  );
}