document.addEventListener('DOMContentLoaded', function() {
    const menuIcon = document.getElementById('menu-icon');
    const navbar = document.querySelector('.navbar');

    menuIcon.addEventListener('click', function() {
        // Toggle the 'active' class on the navbar
        navbar.classList.toggle('active');
    });
});
