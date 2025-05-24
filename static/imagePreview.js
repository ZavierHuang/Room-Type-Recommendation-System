// 圖片預覽與放大功能
document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('roomImagePreview').addEventListener('click', function() {
    var img = document.getElementById('roomImagePreview');
    if (img && img.src && img.style.display !== 'none') {
      var zoomModal = document.getElementById('imageZoomModal');
      var zoomedImg = document.getElementById('zoomedImage');
      if (zoomModal && zoomedImg) {
        zoomedImg.src = img.src;
        zoomModal.style.display = 'flex';
      }
    }
  });
  document.getElementById('closeZoomModal').addEventListener('click', function() {
    document.getElementById('imageZoomModal').style.display = 'none';
  });
  document.getElementById('imageZoomModal').addEventListener('click', function(e) {
    if (e.target === this) {
      this.style.display = 'none';
    }
  });
});

