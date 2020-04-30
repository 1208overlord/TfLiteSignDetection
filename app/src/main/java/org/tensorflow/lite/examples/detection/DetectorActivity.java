/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.location.Address;
import android.location.Geocoder;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.io.IOException;
import java.text.NumberFormat;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import static android.location.Geocoder.isPresent;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 512;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "speedsign.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/speedsign.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.8f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  TextView tvSpeed;
  TextView tvAddress;

  String prev_sign_text = "";
  boolean showToast = false;

  TextView tv_1;
  TextView tv_2;
  TextView tv_3;
  TextView tv_4;

  DetectorActivity activity;




  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);

    tvSpeed = findViewById(R.id.tvSpeed);
    tvAddress = findViewById(R.id.tvAddress);

    tv_1 = findViewById(R.id.tv_1);
    tv_2 = findViewById(R.id.tv_2);
    tv_3 = findViewById(R.id.tv_3);
    tv_4 = findViewById(R.id.tv_4);

    activity = this;

    if (!this.isLocationEnabled(this)) {


      //show dialog if Location Services is not enabled


      AlertDialog.Builder builder = new AlertDialog.Builder(this);
      builder.setTitle("GPS Not Enabled");  // GPS not found
      builder.setMessage("This app requires GPS or Location Service.\\n\\nWould you like to enable Location Service now?\\n"); // Want to enable?
      builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
        public void onClick(DialogInterface dialogInterface, int i) {

          Intent intent = new Intent(android.provider.Settings.ACTION_LOCATION_SOURCE_SETTINGS);
          intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);

          activity.startActivity(intent);
        }
      });

      //if no - bring user to selecting Static Location Activity
      builder.setNegativeButton("No", new DialogInterface.OnClickListener() {

        @Override
        public void onClick(DialogInterface dialog, int which) {
          Toast.makeText(activity, "Please enable Location-based service / GPS", Toast.LENGTH_LONG).show();


        }


      });
      builder.create().show();


    }

    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);


    new SpeedTask(this).execute("string");

  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);

                LayoutInflater inflater = getLayoutInflater();
                View view = inflater.inflate(R.layout.signtoast_80,
                        (ViewGroup)findViewById(R.id.speed_sign_80));
                switch (result.getTitle()) {
                  case "100":
                    view = inflater.inflate(R.layout.signtoast_100,
                            (ViewGroup) findViewById(R.id.speed_sign_100));
                    break;
                  case "120":
                    view = inflater.inflate(R.layout.signtoast_120,
                            (ViewGroup) findViewById(R.id.speed_sign_120));
                    break;
                  case "90":
                    view = inflater.inflate(R.layout.signtoast_90,
                            (ViewGroup) findViewById(R.id.speed_sign_90));
                    break;
                }
                Toast toast = new Toast(getApplicationContext());
                toast.setView(view);
                if(!result.getTitle().equals("") && !prev_sign_text.equals(result.getTitle())){
                  toast.show();
                  showToast = false;

                  tv_4.setText(tv_3.getText());
                  tv_3.setText(tv_2.getText());
                  tv_2.setText(tv_1.getText());
                  tv_1.setText(result.getTitle());

                  if(!tv_1.getText().equals(""))
                    tv_1.setBackgroundResource(R.drawable.ring_background_1);
                  if(!tv_2.getText().equals(""))
                    tv_2.setBackgroundResource(R.drawable.ring_background_1);
                  if(!tv_3.getText().equals(""))
                    tv_3.setBackgroundResource(R.drawable.ring_background_1);
                  if(!tv_4.getText().equals(""))
                    tv_4.setBackgroundResource(R.drawable.ring_background_1);
                }




                // Check database on firebase
                FirebaseDatabase database = FirebaseDatabase.getInstance();
//                if(((String) tvAddress.getText()).equals(""))
//                  continue;
                DatabaseReference mDatabase = database.getReference();
                if(prev_sign_text.equals("") || !prev_sign_text.equals(result.getTitle()))
                  mDatabase.child("street_limit_speed").child((String) tvAddress.getText()).setValue(result.getTitle());

                // Read from the database
                mDatabase.addValueEventListener(new ValueEventListener() {
                  @Override
                  public void onDataChange(DataSnapshot dataSnapshot) {
                    // This method is called once with the initial value and again
                    // whenever data at this location is updated.
                    String value = dataSnapshot.getValue(String.class);
                  }

                  @Override
                  public void onCancelled(DatabaseError error) {
                    // Failed to read value
                    Log.w("FireBase", "Failed to read value.", error.toException());
                  }
                });

                prev_sign_text = result.getTitle();
                showToast = true;
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @SuppressLint("ResourceType")
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }

  private class SpeedTask extends AsyncTask<String, Void, String> {
    final DetectorActivity activity;
    float speed = 0.0f;
    double lat;
    LocationManager locationManager;

    public SpeedTask(DetectorActivity activity) {
      this.activity = activity;
    }

    @Override
    protected String doInBackground(String... params) {
      locationManager = (LocationManager) activity.getSystemService(Context.LOCATION_SERVICE);


      return null;

    }

    protected void onPostExecute(String result) {
      tvSpeed.setText("Km/h");
      LocationListener listener = new LocationListener() {
        float filtSpeed;
        float localspeed;

        @Override
        public void onLocationChanged(Location location) {
          speed = location.getSpeed();
          float multiplier = 3.6f;

          localspeed = speed * multiplier;

          filtSpeed = filter(filtSpeed, localspeed, 2);


          NumberFormat numberFormat = NumberFormat.getNumberInstance();
          numberFormat.setMaximumFractionDigits(0);


          lat = location.getLatitude();
          tvSpeed.setText(numberFormat.format(filtSpeed) + " Km/h");

          numberFormat.setMaximumFractionDigits(0);
          NumberFormat nf = NumberFormat.getInstance();
          nf.setMaximumFractionDigits(4);
          Geocoder geocoder = new Geocoder(activity, Locale.getDefault());
          boolean isGecoder = Geocoder.isPresent();


          try {
            List<Address> addresses = geocoder.getFromLocation(location.getLatitude(), location.getLongitude(), 1);

            if (addresses != null) {
              Address returnedAddress = addresses.get(0);
              StringBuilder strReturnedAddress = new StringBuilder();
              for (int i = 0; i < returnedAddress.getMaxAddressLineIndex(); i++) {
                strReturnedAddress.append(returnedAddress.getAddressLine(i)).append("");
              }
              String street = returnedAddress.getThoroughfare();
              tvAddress.setText("Address: " + isGecoder + strReturnedAddress.length() + ":" + strReturnedAddress.toString());
              tvAddress.setText("Street: " + street);
            }
            else {
              tvAddress.setText("No Address returned!");
            }
          } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            tvAddress.setText("Canont get Address!");
          }


        }

        @Override
        public void onStatusChanged(String provider, int status, Bundle extras) {
          // TODO Auto-generated method stub

        }

        @Override
        public void onProviderEnabled(String provider) {
//          tvSpeed.setText("STDBY");
//          tvMaxSpeed.setText("NIL");
//
//          tvLat.setText("LATITUDE");
//          tvLon.setText("LONGITUDE");
//          tvHeading.setText("HEADING");
//          tvAccuracy.setText("ACCURACY");

        }

        @Override
        public void onProviderDisabled(String provider) {
//          tvSpeed.setText("NOFIX");
//          tvMaxSpeed.setText("NOGPS");
//          tvLat.setText("LATITUDE");
//          tvLon.setText("LONGITUDE");
//          tvHeading.setText("HEADING");
//          tvAccuracy.setText("ACCURACY");


        }

      };

           if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                // TODO: Consider calling
                //    ActivityCompat#requestPermissions
                // here to request the missing permissions, and then overriding
                //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                //                                          int[] grantResults)
                // to handle the case where the user grants the permission. See the documentation
                // for ActivityCompat#requestPermissions for more details.
                return;
            }
           locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0, 0, listener);



    }

    /**
     * Simple recursive filter
     *
     * @param prev Previous value of filter
     * @param curr New input value into filter
     * @return New filtered value
     */
    private float filter(final float prev, final float curr, final int ratio) {
      // If first time through, initialise digital filter with current values
      if (Float.isNaN(prev))
        return curr;
      // If current value is invalid, return previous filtered value
      if (Float.isNaN(curr))
        return prev;
      // Calculate new filtered value
      return (float) (curr / ratio + prev * (1.0 - 1.0 / ratio));
    }


  }

  private boolean isLocationEnabled(Context mContext) {


    LocationManager locationManager = (LocationManager)
            mContext.getSystemService(Context.LOCATION_SERVICE);
    return locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);
  }

}
