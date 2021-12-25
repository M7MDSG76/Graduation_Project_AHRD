
Dropzone.autoDiscover = false;

const myDropzone = new Dropzone("#my-dropzone", {
    url:'up',
    maxFiles: 10,
    maxFilesize: 2,
    acceptedFiles: '.jpg, .png, .tiff',
});
