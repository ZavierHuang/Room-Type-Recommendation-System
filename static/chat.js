$(document).ready(function() {
    $('#chat-header').click(function() {
        $('#chat-sidebar').addClass('collapsed');
        $('#chat-toggle-btn').show();
        $('.container').removeClass('with-chat');
    });

    $('#chat-toggle-btn').click(function() {
        $('#chat-sidebar').removeClass('collapsed');
        $('#chat-toggle-btn').hide();
        $('.container').addClass('with-chat');
    });

    $('#send-btn').click(sendMessage);
    $('#chat-input').keypress(function(e) {
        if (e.which === 13) sendMessage();
    });

    function getCurrentTime() {
        const now = new Date();
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    function sendMessage() {
        let message = $('#chat-input').val();
        if (message.trim() === '') return;

        const timestamp = getCurrentTime();

        // 使用者訊息 + 左下角時間戳
        $('#chat-messages').append(`
            <div class="chat-message user">
                <div class="chat-bubble user d-flex flex-column">
                    <div>${message}</div>
                    <div class="text-start text-muted mt-1" style="font-size: 0.7rem;">${timestamp}</div>
                </div>
            </div>
        `);
        $('#chat-input').val('');

        // 加入 loading 卡片
        const loadingId = `loading-${Date.now()}`;
        $('#chat-messages').append(`
            <div id="${loadingId}" class="chat-message ai d-flex flex-column align-items-start mb-2">
                <div class="d-flex align-items-center">
                    <img src="/static/ai-avatar.png" class="chat-avatar me-2">
                    <div class="chat-bubble ai d-flex align-items-center">
                        <div class="spinner-border spinner-border-sm text-primary me-2" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        正在理解您的問題...
                    </div>
                </div>
            </div>
        `);

        $.ajax({
            url: '/chat',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ message: message }),
            success: function(data) {
                const response = data.response;
                const rooms = response.rooms;
                const conclusion = response.conclusion;

                console.log("response:", response)
                console.log("rooms:", rooms)
                console.log("conclusion:", conclusion)

                const aiTimestamp = getCurrentTime();

                // 移除 loading 卡片
                $(`#${loadingId}`).remove();

                // AI 結論訊息 + 右下角時間戳
                $('#chat-messages').append(`
                    <div class="chat-message ai">
                        <img src="/static/ai-avatar.png" class="chat-avatar">
                        <div class="chat-bubble ai d-flex flex-column">
                            <div>${conclusion.replace(/\n/g, '<br>')}</div>
                            <div class="text-end text-muted mt-1" style="font-size: 0.7rem;">${aiTimestamp}</div>
                        </div>
                    </div>
                `);

                if (rooms && Object.keys(rooms).length > 0) {
                    Object.entries(rooms).forEach(([name, room]) => {
                        const roomTimestamp = getCurrentTime();
                        const cardHtml = `
                            <div class="chat-message ai">
                                <img src="/static/ai-avatar.png" class="chat-avatar">
                                <div class="card mb-2" style="width: 100%;">
                                    <div class="card-body d-flex flex-column">
                                        <h5 class="card-title">${name}</h5>
                                        <h6 class="card-subtitle mb-2 text-muted">價格：${room.price} 元</h6>
                                        <p class="card-text">
                                            面積：${room.area}<br>
                                            特色：${room.features}<br>
                                            風格：${room.style}<br>
                                            床數：${room.maxOccupancy}
                                        </p>
                                        <div class="text-end text-muted mt-1" style="font-size: 0.7rem;">${roomTimestamp}</div>
                                    </div>
                                </div>
                            </div>
                        `;
                        $('#chat-messages').append(cardHtml);
                    });
                }

                $('#chat-messages').scrollTop($('#chat-messages')[0].scrollHeight);
            },
            error: function() {
                $(`#${loadingId}`).remove();
                const aiTimestamp = getCurrentTime();
                $('#chat-messages').append(`
                    <div class="chat-message ai">
                        <img src="/static/ai-avatar.png" class="chat-avatar">
                        <div class="chat-bubble ai d-flex flex-column">
                            <div>抱歉，伺服器出現問題，請稍後再試。</div>
                            <div class="text-end text-muted mt-1" style="font-size: 0.7rem;">${aiTimestamp}</div>
                        </div>
                    </div>
                `);
            }
        });
    }
});
