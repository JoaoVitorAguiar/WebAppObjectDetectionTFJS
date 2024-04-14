/**
 * Função para cortar uma imagem com base nas coordenadas de uma bounding box.
 * @param {HTMLImageElement} img - A imagem a ser cortada.
 * @param {Object} box - As coordenadas da bounding box {x, y, width, height}.
 * @returns {HTMLCanvasElement} - A imagem cortada em um canvas.
 */
export const cropImage = (img, box) => {
    // Criar um canvas com base nas dimensões da bounding box
    const canvas = document.createElement('canvas');
    canvas.width = box.width;
    canvas.height = box.height;
    const ctx = canvas.getContext('2d');

    // Desenhar a parte da imagem especificada pela bounding box no canvas
    ctx.drawImage(
        img,
        box.x, box.y, box.width, box.height, // Coordenadas da bounding box na imagem original
        0, 0, box.width, box.height // Desenhar na origem do canvas
    );

    return canvas;
}