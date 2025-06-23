### Deployment Notes and Compatibility Recommendations for Jetson Nano

1. **JetPack Version Compatibility**
   JetPack 4.2 was tested multiple times but failed to successfully support deep learning frameworks due to incompatibility with TensorFlow versions. It is **recommended to use JetPack 4.6**, which is more stable and supports Python 3.6.9.

   > Installation guidance can be found at: [Official TensorFlow for Jetson Nano](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770)

2. **Limitations of Development Tools**
   PyCharm IDE was tested under JetPack 4.6 but encountered several limitations in functionality. These may stem from configuration issues or hardware constraints. Use lightweight editors or remote development environments for best results.

3. **Avoid Using scikit-learn**
   It is **not recommended to use the `scikit-learn` package** on Jetson Nano, including tools such as `StandardScaler` and `RandomForestClassifier`. The main issue lies in poor support for the `aarch64` architecture, and most online solutions do not work reliably.

4. **EMD-signal Library Compatibility**
   The latest version of the `EMD-signal` library only supports Python 3.7 or later. In the current JetPack 4.6 environment, version 1.2.1 is usable and recommended. Ensure your code does not rely on newer features to avoid compatibility issues.

5. **Data Format Requirements**
   Do **not use `.xlsx` files** in your workflow. All data tables should be stored in **`.csv` format** to ensure compatibility and efficient data loading on Jetson Nano.

---
