from src.pipeline.prediction_pipeline import PredictionPipeline

def test_image_prediction():
    pipeline = PredictionPipeline()
    preds = pipeline.predict_image("sample_img1.jpg")
    print("Image Predictions:", preds)

def test_video_prediction():
    pipeline = PredictionPipeline()
    preds = pipeline.predict_video("sample_video1.mp4")

    count = 0
    for p in preds:
        print("Video Prediction:", p)
        count += 1
        if count >= 5:  # sirf pehle 5 detections print karo
            break


if __name__ == "__main__":
    print("=== Running Prediction Tests ===")
    test_image_prediction()
    test_video_prediction()
