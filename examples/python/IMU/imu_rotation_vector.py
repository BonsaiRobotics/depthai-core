#!/usr/bin/env python3
from collections import deque
import math
import time

import depthai as dai


WINDOW_SECONDS = 10.0
PRINT_UPDATE_HZ = 4.0
YZ_SWAP_ROTATION = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
]


def quaternion_to_euler_xyz(i: float, j: float, k: float, real: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (real * i + j * k)
    cosr_cosp = 1.0 - 2.0 * (i * i + j * j)
    x = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (real * j - k * i)
    sinp = max(-1.0, min(1.0, sinp))
    y = math.degrees(math.asin(sinp))

    siny_cosp = 2.0 * (real * k + i * j)
    cosy_cosp = 1.0 - 2.0 * (j * j + k * k)
    z = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return x, y, z


def resolve_imu_extrinsics_destination(eeprom_data: dai.EepromData) -> dai.CameraBoardSocket:
    socket = eeprom_data.imuExtrinsics.toCameraSocket
    if socket != dai.CameraBoardSocket.AUTO:
        return socket

    if dai.CameraBoardSocket.CAM_A in eeprom_data.cameraData:
        return dai.CameraBoardSocket.CAM_A

    if eeprom_data.cameraData:
        return next(iter(eeprom_data.cameraData))

    return dai.CameraBoardSocket.CAM_A


def rotation_vector_supported(device: dai.Device) -> bool:
    return device.getConnectedIMU() != "BMI270"


def window_spans(samples: deque[tuple[float, float, float, float, float, float, float, float, float]]) -> tuple[float, float, float]:
    x_values = [sample[1] for sample in samples]
    y_values = [sample[2] for sample in samples]
    z_values = [sample[3] for sample in samples]
    return (
        max(x_values) - min(x_values),
        max(y_values) - min(y_values),
        max(z_values) - min(z_values),
    )


with dai.Pipeline() as pipeline:
    device = pipeline.getDefaultDevice()
    imu_name = device.getConnectedIMU()
    if not rotation_vector_supported(device):
        print(f"Skipping example: connected IMU {imu_name} does not support ROTATION_VECTOR.")
        raise SystemExit(0)

    # Define sources and outputs
    imu = pipeline.create(dai.node.IMU)

    # Enable ROTATION_VECTOR at 200 hz rate
    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 200)
    # It's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
    # Above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu.setBatchReportThreshold(1)
    # Maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    # If lower or equal to batchReportThreshold then the sending is always blocking on device
    # Useful to reduce device's CPU load and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu.setMaxBatchReports(10)

    imuQueue = imu.out.createOutputQueue(maxSize=50, blocking=False)
    calibration = device.readCalibration()
    eeprom_data = calibration.getEepromData()
    imu_extrinsics = eeprom_data.imuExtrinsics

    destination_socket = resolve_imu_extrinsics_destination(eeprom_data)
    translation = [imu_extrinsics.translation.x, imu_extrinsics.translation.y, imu_extrinsics.translation.z]
    spec_translation = [imu_extrinsics.specTranslation.x, imu_extrinsics.specTranslation.y, imu_extrinsics.specTranslation.z]

    calibration.setImuExtrinsics(destination_socket, YZ_SWAP_ROTATION, translation, spec_translation)
    device.setCalibration(calibration)

    pipeline.start()
    start_time = time.monotonic()
    last_print_time = 0.0
    samples: deque[tuple[float, float, float, float, float, float, float, float, float]] = deque()

    print(f"Applied runtime IMU extrinsics Y/Z swap relative to {destination_socket.name}.")
    print(f"Rotation matrix: {YZ_SWAP_ROTATION}")
    print(f"Rotation vector stream started on IMU {imu_name}.")
    print("Move the device around each axis and watch the rolling XYZ angle spans respond.")
    print("Expected use: X/Y/Z spans should match the physical axis you rotate around, without swapped signs.")

    while pipeline.isRunning():
        try:
            imuData = imuQueue.get()
        except KeyboardInterrupt:
            break

        assert isinstance(imuData, dai.IMUData)

        for imuPacket in imuData.packets:
            rotationVector = imuPacket.rotationVector
            x_angle, y_angle, z_angle = quaternion_to_euler_xyz(
                rotationVector.i,
                rotationVector.j,
                rotationVector.k,
                rotationVector.real,
            )

            sample_time = time.monotonic() - start_time
            samples.append(
                (
                    sample_time,
                    x_angle,
                    y_angle,
                    z_angle,
                    rotationVector.i,
                    rotationVector.j,
                    rotationVector.k,
                    rotationVector.real,
                    rotationVector.rotationVectorAccuracy,
                )
            )

            while samples and sample_time - samples[0][0] > WINDOW_SECONDS:
                samples.popleft()

        current_time = time.monotonic()
        if samples and current_time - last_print_time >= 1.0 / PRINT_UPDATE_HZ:
            latest = samples[-1]
            x_span, y_span, z_span = window_spans(samples)
            print(
                f"Window {WINDOW_SECONDS:4.1f}s span [deg]: x={x_span:7.2f} y={y_span:7.2f} z={z_span:7.2f} | "
                f"latest xyz=[{latest[1]:7.2f}, {latest[2]:7.2f}, {latest[3]:7.2f}] "
                f"quat=[{latest[4]: .5f}, {latest[5]: .5f}, {latest[6]: .5f}, {latest[7]: .5f}] "
                f"acc={latest[8]:.5f} rad"
            )
            last_print_time = current_time
