import { Camera, CameraType } from "expo-camera";
import { useState, useEffect, useRef } from "react";
import {
  Button,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Pressable,
} from "react-native";
import * as MediaLibrary from "expo-media-library"; // Ensure you've imported MediaLibrary

export default function App() {
  const [type, setType] = useState(CameraType.back);
  const [hasPermission, setHasPermission] = useState(null);
  const [sequence, setSequence] = useState(null); // state to hold the URI of recorded video
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const mediaLibraryPermissions =
        await MediaLibrary.requestPermissionsAsync();
      const cameraStatus = await Camera.requestCameraPermissionsAsync();
      const soundStatus = await Camera.requestMicrophonePermissionsAsync();
      setHasPermission(
        cameraStatus.granted &&
          soundStatus.granted &&
          mediaLibraryPermissions.granted
      );
    })();
  }, []);

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text>No access to camera or microphone</Text>
      </View>
    );
  }

  if (hasPermission === null) {
    // Permissions are still loading
    return (
      <View style={styles.container}>
        <Text>Loading permissions...</Text>
      </View>
    );
  }

  const takeVideo = async () => {
    if (cameraRef.current) {
      try {
        const data = await cameraRef.current.recordAsync({ quality: "4:3" });
        setSequence(data.uri);
      } catch (e) {
        console.error(e);
      }
    }
  };

  const toggleCameraType = () => {
    setType((prevType) =>
      prevType === CameraType.back ? CameraType.front : CameraType.back
    );
  };

  return (
    <View style={styles.container}>
      <Camera
        style={styles.camera}
        type={type}
        ref={cameraRef}
        flashMode={auto}
      >
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraType}>
            <Text style={styles.text}>Flip Camera</Text>
          </TouchableOpacity>
        </View>
      </Camera>
      <Pressable onPress={takeVideo} style={styles.pressable}>
        <Text style={styles.text}>
          {sequence ? "Stop Recording" : "Start Recording"}
        </Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  camera: {
    flex: 1,
    width: "100%",
  },
  buttonContainer: {
    flex: 0.1,
    flexDirection: "row",
    backgroundColor: "transparent",
    margin: 10,
    justifyContent: "space-between",
  },
  button: {
    flex: 0.3,
    alignSelf: "flex-end",
    alignItems: "center",
    padding: 10,
  },
  pressable: {
    padding: 10,
    backgroundColor: "#333",
    margin: 10,
  },
  text: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
  },
});
