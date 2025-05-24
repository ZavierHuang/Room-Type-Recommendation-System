// 房型表單相關功能
function checkRoomImageBtnEnable() {
  var requiredFields = [
    'roomName',
    'roomPrice',
    'roomArea',
    'roomFeatures',
    'roomStyle',
    'roomMaxOccupancy'
  ];
  var enable = true;
  for (var i = 0; i < requiredFields.length; i++) {
    var el = document.getElementById(requiredFields[i]);
    if (!el || el.value === '' || el.value === null) {
      enable = false;
      break;
    }
  }
  document.getElementById('roomImageBtn').disabled = !enable;
}

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('autoFillBtn').addEventListener('click', function() {
    document.getElementById('roomName').value = '';
    document.getElementById('roomPrice').value = '';
    document.getElementById('roomArea').value = '';
    document.getElementById('roomFeatures').value = '';
    document.getElementById('roomStyle').value = '';
    document.getElementById('roomMaxOccupancy').value = '';
    var pendingCircle = document.getElementById('pendingCircle');
    if (pendingCircle) pendingCircle.style.display = 'flex';
    fetch('/auto_recommend')
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          if (pendingCircle) pendingCircle.style.display = 'none';
          alert(data.error);
          return;
        }
        document.getElementById('roomFeatures').value = Array.isArray(data.features) ? data.features.join('、') : (data.features || '');
        document.getElementById('roomName').value = data.name || '';
        document.getElementById('roomPrice').value = data.price || '';
        document.getElementById('roomArea').value = data.area || '';
        document.getElementById('roomStyle').value = data.style || '';
        document.getElementById('roomMaxOccupancy').value = data.maxOccupancy || '';
        if (pendingCircle) pendingCircle.style.display = 'none';
        checkRoomImageBtnEnable();
        document.getElementById('roomImageBtn').click();
      });
  });
  document.getElementById('clearAllBtn').addEventListener('click', function() {
    document.getElementById('roomName').value = '';
    document.getElementById('roomPrice').value = '';
    document.getElementById('roomArea').value = '';
    document.getElementById('roomFeatures').value = '';
    document.getElementById('roomStyle').value = '';
    document.getElementById('roomMaxOccupancy').value = '';
    var img = document.getElementById('roomImagePreview');
    var text = document.getElementById('imagePreviewText');
    if (img) {
      img.src = '';
      img.style.display = 'none';
    }
    if (text) {
      text.style.display = 'block';
    }
    checkRoomImageBtnEnable();
  });
  document.getElementById('roomImageBtn').addEventListener('click', function() {
    var pendingCircle = document.getElementById('pendingCircle');
    if (pendingCircle) pendingCircle.style.display = 'flex';
    var data = {
      name: document.getElementById('roomName').value,
      price: document.getElementById('roomPrice').value,
      area: document.getElementById('roomArea').value,
      features: document.getElementById('roomFeatures').value,
      style: document.getElementById('roomStyle').value,
      maxOccupancy: document.getElementById('roomMaxOccupancy').value
    };
    fetch('/generate_room_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
      if (pendingCircle) pendingCircle.style.display = 'none';
      if (result.error) {
        alert(result.error);
        return;
      }
      var img = document.getElementById('roomImagePreview');
      var text = document.getElementById('imagePreviewText');
      if (img && result.image_url) {
        img.src = result.image_url + '?t=' + Date.now();
        img.style.display = 'block';
        if (text) text.style.display = 'none';
      }
    })
    .catch(() => {
      if (pendingCircle) pendingCircle.style.display = 'none';
      alert('圖片生成失敗');
    });
  });
  document.getElementById('addRoomForm').addEventListener('submit', function(e) {
    var img = document.getElementById('roomImagePreview');
    if (!img || !img.src || img.src === '' || img.style.display === 'none') {
      alert('房型圖片尚未生成！');
      e.preventDefault();
      return;
    }
    e.preventDefault();
    var data = {
      name: document.getElementById('roomName').value,
      price: document.getElementById('roomPrice').value,
      area: document.getElementById('roomArea').value,
      features: document.getElementById('roomFeatures').value,
      style: document.getElementById('roomStyle').value,
      maxOccupancy: document.getElementById('roomMaxOccupancy').value,
      image: img.src ? img.src : ''
    };
    fetch('/add_room', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
      if (result.success) {
        alert('新增成功!');
        location.reload();
      } else {
        alert(result.error || '新增失敗');
      }
    })
    .catch(() => alert('新增失敗'));
  });
  ['roomName','roomPrice','roomArea','roomFeatures','roomStyle','roomMaxOccupancy'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', checkRoomImageBtnEnable);
    }
  });
  checkRoomImageBtnEnable();
});

