import sensor
import time
import pyb

ENABLE_LED_TOGGLE = True
BLINK_LED_ID = 2
LED_TOGGLE_EVERY_N_FRAMES = 1

FRAME_MAGIC = b"\xAA\xBB"
JPEG_QUALITY = 85

led = pyb.LED(BLINK_LED_ID)
frame_count = 0

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.XGA)
sensor.skip_frames(time=2000)

usb = pyb.USB_VCP()
usb.setinterrupt(-1)

clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()
    jpeg = img.compress(quality=JPEG_QUALITY)

    usb.write(FRAME_MAGIC)
    usb.write(img.width().to_bytes(2, "little"))
    usb.write(img.height().to_bytes(2, "little"))
    usb.write(len(jpeg).to_bytes(4, "little"))
    usb.write(jpeg)

    frame_count += 1
    if ENABLE_LED_TOGGLE and frame_count % LED_TOGGLE_EVERY_N_FRAMES == 0:
        led.toggle()
