#include "types.h"
// qemu/hw/riscv/virt.c - risc-v memory layout

#define VIRT_VIRTIO 0x10001000

// qemu/include/standard-headers/linux/virtio_mmio.h
/* Control registers */
#define VIRTIO_MMIO_MAGIC_VALUE 0x000         /* Read Only */
#define VIRTIO_MMIO_VERSION 0x004             /* Read Only */
#define VIRTIO_MMIO_DEVICE_ID 0x008           /* Read Only */
#define VIRTIO_MMIO_VENDOR_ID 0x00c           /* Read Only */
#define VIRTIO_MMIO_DEVICE_FEATURES 0x010     /* Read Only */
#define VIRTIO_MMIO_DEVICE_FEATURES_SEL 0x014 /* Write Only */
#define VIRTIO_MMIO_DRIVER_FEATURES 0x020     /* Write Only */
#define VIRTIO_MMIO_DRIVER_FEATURES_SEL 0x024 /* Write Only */
#define VIRTIO_MMIO_QUEUE_SEL 0x030           /* Write Only */
#define VIRTIO_MMIO_STATUS 0x070              /* Read Write */

/* Status register bits */
#define VIRTIO_CONFIG_S_ACKNOWLEDGE 1
#define VIRTIO_CONFIG_S_DRIVER 2
#define VIRTIO_CONFIG_S_DRIVER_OK 4
#define VIRTIO_CONFIG_S_FEATURES_OK 8

/* Feature bits */
#define VIRTIO_BLK_F_RO 5  /* Disk is read-only */
#define VIRTIO_BLK_F_MQ 12 /* Support more than one virtual queue */

#define READ(r) ((volatile uint32 *)(VIRT_VIRTIO + (r)))

// Perform device-specific setup.
// Set the DRIVER_OK status bit to the status register. The device is now LIVE.

void virtio_init() {
        // check magic #, version, device id
        if (*READ(VIRTIO_MMIO_MAGIC_VALUE) != 0x74726976 || *READ(VIRTIO_MMIO_VERSION) != 2 ||
            *READ(VIRTIO_MMIO_DEVICE_ID) != 2) {
                panic("virtio init");
        }
        // reset device
        int status = 0;
        *READ(VIRTIO_MMIO_STATUS) = status;

        // set acknowledge and driver bits in status register
        status |= (VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER);
        *READ(VIRTIO_MMIO_STATUS) = status;

        // negotiate features
        uint64 features = *READ(VIRTIO_MMIO_DEVICE_FEATURES);  // host_features register
        features &= ~(1 << VIRTIO_BLK_F_RO);
        features &= ~(1 << VIRTIO_BLK_F_MQ);
        *READ(VIRTIO_MMIO_DRIVER_FEATURES) = features;  // guest_features register

        // set FEATURES_OK bit in status register
        status |= VIRTIO_CONFIG_S_FEATURES_OK;
        *READ(VIRTIO_MMIO_STATUS) = status;

        // check FEATURES_OK bit - features accepted
        status = *READ(VIRTIO_MMIO_STATUS);
        if (!(status & VIRTIO_CONFIG_S_FEATURES_OK)) panic("VIRTIO_CONFIG_S_FEATURES_OK unset");

        // set up queues

        // set DRIVER_OK - device is live
        status |= VIRTIO_CONFIG_S_DRIVER_OK;
        *READ(VIRTIO_MMIO_STATUS) = status;
}

void virtio_rw() {
        // virtio_blk_req object
        // set up 3 descriptors
        // chain descriptors and attach it to avail array
        // notify device
}